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
        """
        Full skill gap analysis:
        - Scores all jobs by NLP similarity to user skills (title 40% + skills 60%)
        - Returns top 8 matches
        - Finds missing skills per job (up to 6 shown)
        - Recommends up to 3 courses per missing skill (capped at 6 total)
        - Falls back to live DB query if local cache has no match
        """
        if not self.jobs_data:
            return []

        # Normalise input — handle both comma strings and lists
        if isinstance(user_skills_raw, list):
            user_skills_str = ', '.join(user_skills_raw)
        else:
            user_skills_str = _to_str(user_skills_raw)

        user_doc        = nlp(user_skills_str)
        user_skills_set = {
            s.strip().lower()
            for s in user_skills_str.split(',')
            if s.strip()
        }

        # ── Score every cached job ───────────────────────────────────────────
        job_matches = []
        for job in self.jobs_data:
            try:
                title_score  = user_doc.similarity(job['title_doc'])
                skills_score = user_doc.similarity(job['skills_doc'])
                score = (title_score * 0.4) + (skills_score * 0.6)
            except Exception:
                score = 0.0
            job_matches.append({**job, "score": score})

        top_jobs = sorted(job_matches, key=lambda x: x['score'], reverse=True)[:8]

        results = []
        for job in top_jobs:
            # Normalise job skills field — may be a comma string or a list
            raw_skills = job.get('skills', '')
            if isinstance(raw_skills, list):
                job_skills = [s.strip() for s in raw_skills if s.strip()]
            else:
                job_skills = [s.strip() for s in _to_str(raw_skills).split(',') if s.strip()]

            missing = [s for s in job_skills if s.lower() not in user_skills_set]

            # ── Course recommendations per missing skill ──────────────────────
            courses = []
            seen_course_ids = set()

            for skill in missing[:3]:
                skill_lower = skill.lower()

                # Build a searchable text blob for each cached course
                local_hits = []
                for c in self.courses_data:
                    blob = (
                        _to_str(c.get('title'))   + ' ' +
                        _to_str(c.get('skills'))  + ' ' +
                        _to_str(c.get('field'))
                    ).lower()
                    if skill_lower in blob:
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

                # Fallback: live DB query if cache gave nothing for this skill
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
