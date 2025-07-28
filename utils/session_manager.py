from uuid import uuid4

def get_or_create_user_and_session(session_id: str, user_id: str | None):
    from uuid import uuid4
    if not user_id:
        user_id = str(uuid4())
    unique_session_key = f"{session_id}_{user_id}"
    return session_id, user_id, unique_session_key
