from core.shared_utils import nlp


def _to_str(val):
    """Safely convert any field to a plain string for text matching."""
    if isinstance(val, list):
        return ' '.join(str(v) for v in val)
    return str(val) if val else ''


class CareerEngine:
    def __init__(self, supabase_client):
        self.supabase = supabase_client
        self.jobs_data = []
        self.courses_data = []
        self.refresh_cache()

    def refresh_cache(self):
        """Syncs jobs AND courses from Supabase and pre-computes NLP vectors."""
        try:
            print("🔍 Engine: Syncing jobs from Supabase...")
            job_res = self.supabase.table('jobs').select("*").execute()
            raw_jobs = job_res.data or []

            processed = []
            for job in raw_jobs:
                title_text  = _to_str(job.get('title', ''))
                skills_text = _to_str(job.get('skills', ''))
                job['title_doc']  = nlp(title_text)
                job['skills_doc'] = nlp(skills_text)
                processed.append(job)

            self.jobs_data = processed
            print(f"✅ Engine: Cached {len(self.jobs_data)} jobs.")

            print("🔍 Engine: Syncing courses from Supabase...")
            course_res = self.supabase.table('courses').select("*").execute()
            self.courses_data = course_res.data or []
            print(f"✅ Engine: Cached {len(self.courses_data)} courses.")

        except Exception as e:
            print(f"❌ Engine Cache Error: {e}")

    def recommend_by_job(self, user_job_title):
        """Find top matching jobs for a given job title using NLP similarity."""
        if not self.jobs_data:
            return {"error": "No jobs found in cache"}

        user_doc = nlp(_to_str(user_job_title))

        matches = []
        for job in self.jobs_data:
            try:
                score = user_doc.similarity(job['title_doc'])
            except Exception:
                score = 0.0

            matches.append({
                "matched_job": _to_str(job.get('title')),
                "industry":    _to_str(job.get('industry')),
                "accuracy":    round(score * 100, 2),
                "url":         _to_str(job.get('link')) or '#',
                "skills":      _to_str(job.get('skills')),
            })

        return sorted(matches, key=lambda x: x['accuracy'], reverse=True)[0]

   def recommend_by_skills(self, user_skills_raw):
    if not self.jobs_data:
        return []

    if isinstance(user_skills_raw, list):
        user_skills_str = ', '.join(user_skills_raw)
    else:
        user_skills_str = _to_str(user_skills_raw)

    user_skills_set = {
        s.strip().lower()
        for s in user_skills_str.replace(',', ' ').split()
        if len(s.strip()) > 1
    }
    user_doc = nlp(user_skills_str)

    job_matches = []
    for job in self.jobs_data:
        # ── Token overlap score (primary) ────────────────────────────────
        raw_skills = job.get('skills', '')
        if isinstance(raw_skills, list):
            job_skill_tokens = {s.strip().lower() for s in raw_skills if s.strip()}
        else:
            job_skill_tokens = {
                s.strip().lower()
                for s in _to_str(raw_skills).replace(',', ' ').split()
                if s.strip()
            }

        if job_skill_tokens:
            overlap = len(user_skills_set & job_skill_tokens)
            overlap_score = overlap / max(len(job_skill_tokens), 1)
        else:
            overlap_score = 0.0

        # ── NLP similarity (secondary fallback weight) ───────────────────
        try:
            title_score  = user_doc.similarity(job['title_doc'])
            skills_score = user_doc.similarity(job['skills_doc'])
            nlp_score = (title_score * 0.3) + (skills_score * 0.7)
        except Exception:
            nlp_score = 0.0

        # Weighted blend: overlap is more reliable for short skill lists
        if overlap_score > 0:
            final_score = (overlap_score * 0.75) + (nlp_score * 0.25)
        else:
            final_score = nlp_score * 0.4   # penalise zero-overlap jobs heavily

        job_matches.append({**job, "score": final_score})

    top_jobs = sorted(job_matches, key=lambda x: x['score'], reverse=True)[:8]

    results = []
    for job in top_jobs:
        raw_skills = job.get('skills', '')
        if isinstance(raw_skills, list):
            job_skills = [s.strip() for s in raw_skills if s.strip()]
        else:
            job_skills = [s.strip() for s in _to_str(raw_skills).split(',') if s.strip()]

        missing = [s for s in job_skills if s.lower() not in user_skills_set]

        # ── Course matching: token overlap instead of substring only ─────
        courses = []
        seen_course_ids = set()

        for skill in missing[:3]:
            skill_tokens = set(skill.lower().split())

            local_hits = []
            for c in self.courses_data:
                blob = (
                    _to_str(c.get('title'))  + ' ' +
                    _to_str(c.get('skills')) + ' ' +
                    _to_str(c.get('field'))
                ).lower()
                blob_tokens = set(blob.split())
                # Match if any skill token appears in blob
                if skill_tokens & blob_tokens:
                    local_hits.append(c)

            local_hits = local_hits[:20]

            for c in local_hits:
                cid = c.get('id') or c.get('title')
                if cid not in seen_course_ids:
                    seen_course_ids.add(cid)
                    courses.append({
                        "title":    _to_str(c.get('title')),
                        "link":     _to_str(c.get('link') or c.get('url')) or '#',
                        "provider": _to_str(c.get('provider')),
                        "skill":    skill,
                    })
                    if len(courses) >= 6:
                        break

            if not local_hits:
                try:
                    res = self.supabase.table('courses').select("*") \
                        .ilike('title', f'%{skill}%').limit(3).execute()
                    db_courses = res.data or []
                    if not db_courses:
                        res2 = self.supabase.table('courses').select("*") \
                            .ilike('field', f'%{skill}%').limit(3).execute()
                        db_courses = res2.data or []
                    for c in db_courses:
                        cid = c.get('id') or c.get('title')
                        if cid not in seen_course_ids:
                            seen_course_ids.add(cid)
                            courses.append({
                                "title":    _to_str(c.get('title')),
                                "link":     _to_str(c.get('link') or c.get('url')) or '#',
                                "provider": _to_str(c.get('provider')),
                                "skill":    skill,
                            })
                            if len(courses) >= 6:
                                break
                except Exception as e:
                    print(f"⚠️ Course fetch error for '{skill}': {e}")

            if len(courses) >= 6:
                break

        results.append({
            "job":              _to_str(job.get('title')),
            "industry":         _to_str(job.get('industry')),
            "url":              _to_str(job.get('link')) or '#',
            "match_confidence": round(job['score'] * 100, 2),
            "missing_skills":   missing[:6],
            "courses":          courses[:6],
        })

    return results
