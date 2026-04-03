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
                # Pre-computing docs saves massive amounts of time during requests
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
        """Blended algorithm: Token overlap + Spacy NLP Similarity."""
        if not self.jobs_data:
            return []

        # Normalize user skills
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
            # 1. Token Overlap Score
            raw_skills = job.get('skills', '')
            if isinstance(raw_skills, list):
                job_skill_tokens = {s.strip().lower() for s in raw_skills if s.strip()}
            else:
                job_skill_tokens = {
                    s.strip().lower() 
                    for s in _to_str(raw_skills).replace(',', ' ').split() 
                    if s.strip()
                }

            overlap_score = 0.0
            if job_skill_tokens:
                overlap = len(user_skills_set & job_skill_tokens)
                overlap_score = overlap / max(len(job_skill_tokens), 1)

            # 2. NLP Similarity Fallback
            try:
                title_score  = user_doc.similarity(job['title_doc'])
                skills_score = user_doc.similarity(job['skills_doc'])
                nlp_score = (title_score * 0.3) + (skills_score * 0.7)
            except Exception:
                nlp_score = 0.0

            # Weighted blend
            if overlap_score > 0:
                final_score = (overlap_score * 0.75) + (nlp_score * 0.25)
            else:
                final_score = nlp_score * 0.4

            job_matches.append({**job, "score": final_score})

        top_jobs = sorted(job_matches, key=lambda x: x['score'], reverse=True)[:8]

        results = []
        for job in top_jobs:
            # Parse skills for missing list
            raw_skills = job.get('skills', '')
            if isinstance(raw_skills, list):
                job_skills_list = [s.strip() for s in raw_skills if s.strip()]
            else:
                job_skills_list = [s.strip() for s in _to_str(raw_skills).split(',') if s.strip()]

            missing = [s for s in job_skills_list if s.lower() not in user_skills_set]

            # Course matching
            matched_courses = []
            seen_ids = set()

            for skill in missing[:3]:
                skill_tokens = set(skill.lower().split())
                
                # Check cache first
                for c in self.courses_data:
                    blob = f"{c.get('title')} {c.get('skills')} {c.get('field')}".lower()
                    blob_tokens = set(blob.split())
                    
                    if skill_tokens & blob_tokens:
                        cid = c.get('id') or c.get('title')
                        if cid not in seen_ids:
                            seen_ids.add(cid)
                            matched_courses.append({
                                "title": _to_str(c.get('title')),
                                "link": _to_str(c.get('link') or c.get('url')) or '#',
                                "provider": _to_str(c.get('provider')),
                                "skill": skill
                            })
                    if len(matched_courses) >= 6: break

                # DB fallback if cache is empty for this skill
                if len(matched_courses) < 3:
                    try:
                        res = self.supabase.table('courses').select("*").ilike('title', f'%{skill}%').limit(3).execute()
                        for c in (res.data or []):
                            cid = c.get('id') or c.get('title')
                            if cid not in seen_ids:
                                seen_ids.add(cid)
                                matched_courses.append({
                                    "title": _to_str(c.get('title')),
                                    "link": _to_str(c.get('link') or c.get('url')) or '#',
                                    "provider": _to_str(c.get('provider')),
                                    "skill": skill
                                })
                    except: pass

            results.append({
                "job": _to_str(job.get('title')),
                "industry": _to_str(job.get('industry')),
                "url": _to_str(job.get('link')) or '#',
                "match_confidence": round(job['score'] * 100, 2),
                "missing_skills": missing[:6],
                "courses": matched_courses[:6],
            })

        return results
