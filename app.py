from dotenv import load_dotenv
from flask import Flask, request, jsonify
from supabase import create_client, Client
from flask_cors import CORS

from core.engine import CareerEngine
from core.knowledge_base import COURSE_DB

import os
import time
import pandas as pd
import spacy
from datetime import datetime, timedelta, timezone
from core.shared_utils import nlp

nlp = spacy.load("en_core_web_sm")

load_dotenv()

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(url, key)

engine = CareerEngine(supabase)


def normalize_profile(raw):
    if not raw or not isinstance(raw, dict):
        return {}
    return {
        "id": raw.get("id", ""),
        "name": raw.get("full_name", raw.get("name", "")),
        "full_name": raw.get("full_name", ""),
        "username": raw.get("username", ""),
        "email": raw.get("email", ""),
        "role": raw.get("role", "student"),
        "qualification": raw.get("qualification", ""),
        "phone": raw.get("phone", ""),
        "address": raw.get("location", raw.get("address", "")),
        "location": raw.get("location", ""),
        "tenth": raw.get("tenth", ""),
        "twelfth": raw.get("twelfth", ""),
        "graduation": raw.get("graduation", ""),
        "skills": raw.get("skills", []),
        "photo": raw.get("photo", None),
        "about": raw.get("about", ""),
        "certificates": raw.get("certificates", []),
        "personalPosts": raw.get("personal_posts", raw.get("personalPosts", [])),
        "resumes": raw.get("resumes", []),
        "chats": raw.get("chats", {}),
        "company_name": raw.get("company_name", ""),
        "tagline": raw.get("tagline", ""),
        "domain": raw.get("domain", ""),
        "website": raw.get("website", ""),
        "founded": raw.get("founded", ""),
        "achievements": raw.get("achievements", []),
        "created_at": raw.get("created_at", ""),
    }


def _first_row(response):
    data = response.data
    if not data:
        return None
    if isinstance(data, list):
        return data[0] if data else None
    return data


@app.route('/api/get-profile', methods=['GET'])
def get_profile():
    try:
        user_id = request.args.get('user_id')
        if user_id:
            response = supabase.table('profiles').select("*").eq('id', user_id).execute()
        else:
            response = supabase.table('profiles').select("*").limit(1).execute()

        row = _first_row(response)
        if not row:
            return jsonify({"error": "No profile found"}), 404

        return jsonify(normalize_profile(row))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/profile/<user_id>', methods=['GET'])
def get_profile_by_id(user_id):
    try:
        response = supabase.table('profiles').select("*").eq('id', user_id).execute()
        row = _first_row(response)
        if not row:
            return jsonify({"error": "Profile not found"}), 404
        return jsonify(normalize_profile(row))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/profile/<user_id>', methods=['PUT'])
def update_profile(user_id):
    try:
        updates = request.json
        if not updates:
            return jsonify({"error": "No data provided"}), 400

        db_updates = {}
        field_map = {
            "name": "full_name", "fullName": "full_name", "address": "location",
            "personalPosts": "personal_posts", "graduation": "graduation"
        }
        
        for key, val in updates.items():
            mapped_key = field_map.get(key, key)

            # --- FORCED SKILLS FIX ---
            if mapped_key == "skills":
                if val is None:
                    db_updates["skills"] = []
                elif isinstance(val, list):
                    # Clean the list of empty strings and duplicates
                    seen = set()
                    db_updates["skills"] = [str(s).strip() for s in val if str(s).strip() and not (str(s).strip() in seen or seen.add(str(s).strip()))]
                else:
                    db_updates["skills"] = [s.strip() for s in str(val).split(",") if s.strip()]
                continue 

            db_updates[mapped_key] = val

        # SAFETY: Use .execute() instead of .single() to prevent crashes on empty results
        response = supabase.table('profiles').update(db_updates).eq('id', user_id).execute()
        
        # Check if we actually got data back using the helper
        row = _first_row(response)
        if not row:
            return jsonify({"error": "Profile not found or update failed"}), 404
            
        return jsonify(normalize_profile(row))
    except Exception as e:
        print(f"Backend Error: {str(e)}") # This will show in your Render logs
        return jsonify({"error": str(e)}), 500
        

@app.route('/api/users', methods=['GET'])
def get_users():
    try:
        role = request.args.get('role')
        query = supabase.table('profiles').select("*")
        if role:
            query = query.eq('role', role)
        response = query.execute()
        return jsonify([normalize_profile(u) for u in (response.data or [])])
    except Exception as e:
        return jsonify([])


@app.route('/api/vacancies', methods=['GET'])
def get_vacancies():
    try:
        response = supabase.table('vacancies').select("*").order('created_at', desc=True).execute()
        return jsonify(response.data or [])
    except Exception as e:
        return jsonify([])


@app.route('/api/vacancies', methods=['POST'])
def create_vacancy():
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No data provided"}), 400

        title = data.get("title", "").strip()
        if not title:
            return jsonify({"error": "title is required"}), 400

        owner_id = data.get("owner_id")
        if not owner_id:
            return jsonify({"error": "owner_id is required — user may not be authenticated"}), 400

        desc = data.get("desc", data.get("description", "")).strip()
        if not desc:
            return jsonify({"error": "description is required"}), 400

        payload = {
            "title":       title,
            "description": desc,
            "type":        data.get("type", "Job Vacancy"),
            "skills":      data.get("skills", ""),
            "duration":    data.get("duration", ""),
            "offerings":   data.get("offerings", ""),
            "location":    data.get("location", ""),
            "owner_id":    owner_id,
            "owner_name":  data.get("owner_name", ""),
        }

        response = supabase.table('vacancies').insert(payload).execute()
        row = _first_row(response)
        if not row:
            return jsonify({"error": "Insert failed — no row returned"}), 500
        return jsonify(row), 201

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/vacancies/<vacancy_id>', methods=['DELETE'])
def delete_vacancy(vacancy_id):
    try:
        supabase.table('vacancies').delete().eq('id', vacancy_id).execute()
        return jsonify({"deleted": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/applications/vacancy/<vacancy_id>', methods=['GET'])
def get_vacancy_applications(vacancy_id):
    try:
        response = (
            supabase.table('applications')
            .select("*, profiles(full_name, email)")
            .eq('vacancy_id', vacancy_id)
            .order('created_at', desc=True)
            .execute()
        )
        apps = []
        for a in (response.data or []):
            profile = a.get('profiles') or {}
            apps.append({
                "id":           a.get('id'),
                "student_id":   a.get('student_id'),
                "student_name": profile.get('full_name') or a.get('student_name', 'Student'),
                "email":        profile.get('email') or a.get('email', ''),
                "cover_letter": a.get('cover_letter', ''),
                "status":       a.get('status', 'Pending'),
                "created_at":   a.get('created_at', ''),
            })
        return jsonify(apps)
    except Exception as e:
        return jsonify([])


@app.route('/api/applications', methods=['POST'])
def create_application():
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No data provided"}), 400
        response = supabase.table('applications').insert(data).execute()
        row = _first_row(response)
        if not row:
            return jsonify({"error": "Insert failed"}), 500
        return jsonify(row), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/applications/student/<student_id>', methods=['GET'])
def get_student_applications(student_id):
    try:
        response = (
            supabase.table('applications')
            .select("*, vacancies(*)")
            .eq('student_id', student_id)
            .execute()
        )
        return jsonify(response.data or [])
    except Exception as e:
        return jsonify([])


@app.route('/api/applications/<int:app_id>/status', methods=['PUT'])
def update_application_status(app_id):
    try:
        new_status = request.json.get("status", "Pending")
        response = (
            supabase.table('applications')
            .update({"status": new_status})
            .eq('id', app_id)
            .execute()
        )
        row = _first_row(response)
        if not row:
            return jsonify({"error": "Application not found"}), 404
        return jsonify(row)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/messages/<user_id>', methods=['GET'])
def get_messages(user_id):
    try:
        sent = supabase.table('messages').select("*").eq('sender_id', user_id).execute()
        received = supabase.table('messages').select("*").eq('receiver_id', user_id).execute()
        all_msgs = (sent.data or []) + (received.data or [])
        all_msgs.sort(key=lambda m: m.get('created_at', ''))
        return jsonify(all_msgs)
    except Exception as e:
        return jsonify([])


@app.route('/api/messages', methods=['POST'])
def send_message():
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No data provided"}), 400
        response = supabase.table('messages').insert(data).execute()
        row = _first_row(response)
        if not row:
            return jsonify({"error": "Insert failed"}), 500
        return jsonify(row), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/all-jobs', methods=['GET'])
def get_jobs():
    try:
        res = supabase.table('jobs').select("*").order('last_seen', desc=True).execute()
        return jsonify(res.data or [])
    except Exception as e:
        return jsonify([])


@app.route('/api/analyze-job', methods=['POST'])
def analyze_job():
    title = request.json.get('job_title', '')
    return jsonify(engine.recommend_by_job(title))


@app.route('/api/analyze-skills', methods=['POST'])
def analyze():
    skills = request.json.get('skills', '')
    return jsonify(engine.recommend_by_skills(skills))


@app.route('/api/industries', methods=['GET'])
def get_industries():
    try:
        response = supabase.table('profiles').select("*").eq('role', 'industry').execute()
        result = []
        for ind in (response.data or []):
            result.append({
                "id":       ind.get("id"),
                "name":     ind.get("company_name") or ind.get("full_name"),
                "logo":     (ind.get("company_name") or "C")[:2].upper(),
                "domain":   ind.get("domain", ""),
                "location": ind.get("location", ""),
                "tagline":  ind.get("tagline", ""),
            })
        return jsonify(result)
    except Exception as e:
        return jsonify([])


@app.route('/api/courses', methods=['GET'])
def get_courses():
    try:
        qual = request.args.get('qualification')
        query = supabase.table('courses').select("*")
        if qual:
            query = query.ilike('field', f'%{qual}%')
        response = query.execute()
        return jsonify(response.data or [])
    except Exception as e:
        return jsonify([])


if __name__ == '__main__':
    port = int(os.getenv("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
