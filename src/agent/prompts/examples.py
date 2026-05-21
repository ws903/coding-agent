PLANNER_EXAMPLES = [
    {
        "role": "user",
        "content": (
            "## Task\n"
            "Add a health check endpoint to the Flask app\n\n"
            "## Project Context\n"
            "## File Tree\n"
            "  app.py\n  requirements.txt\n  tests/\n  tests/test_app.py"
        ),
    },
    {
        "role": "assistant",
        "content": (
            "## Plan: Add /health endpoint returning JSON status\n\n"
            "### Step 1: Add the health check route to app.py\n"
            "- Files needed: app.py\n"
            '- Verify: python -c "import app"\n\n'
            "### Step 2: Add a test for the health endpoint\n"
            "- Files needed: tests/test_app.py\n"
            "- Verify: pytest tests/test_app.py -q\n"
        ),
    },
    {
        "role": "user",
        "content": (
            "## Task\n"
            "What database does this project use?\n\n"
            "## Project Context\n"
            "## File Tree\n"
            "  app.py\n  models.py\n  requirements.txt\n  db/schema.sql"
        ),
    },
    {
        "role": "assistant",
        "content": (
            "## Answer\n\n"
            "Based on the file tree, the project uses SQLite. The presence of "
            "db/schema.sql suggests raw SQL schema definitions rather than an ORM. "
            "I would need to see requirements.txt and models.py to confirm whether "
            "an ORM like SQLAlchemy is also in use."
        ),
    },
]

EXECUTOR_EXAMPLES = [
    {
        "role": "user",
        "content": (
            "## Step\n"
            "Add the health check route to app.py\n\n"
            "## Current File Contents\n"
            "### app.py\n"
            "```\n"
            "from flask import Flask, jsonify\n\n"
            "app = Flask(__name__)\n\n"
            "@app.route('/')\n"
            "def index():\n"
            "    return jsonify(message='hello')\n"
            "```"
        ),
    },
    {
        "role": "assistant",
        "content": (
            "Adding a /health route that returns status and 200 OK.\n\n"
            "app.py\n"
            "<<<<<<< SEARCH\n"
            "@app.route('/')\n"
            "def index():\n"
            "    return jsonify(message='hello')\n"
            "=======\n"
            "@app.route('/')\n"
            "def index():\n"
            "    return jsonify(message='hello')\n\n\n"
            "@app.route('/health')\n"
            "def health():\n"
            "    return jsonify(status='ok'), 200\n"
            ">>>>>>> REPLACE"
        ),
    },
    {
        "role": "user",
        "content": (
            "## Step\n"
            "Add a test for the health endpoint\n\n"
            "## Current File Contents\n"
            "### tests/test_app.py\n"
            "(file does not exist yet)"
        ),
    },
    {
        "role": "assistant",
        "content": (
            "Creating a test file for the health endpoint.\n\n"
            "CREATE tests/test_app.py\n"
            "```\n"
            "from app import app\n\n\n"
            "def test_health():\n"
            "    client = app.test_client()\n"
            "    resp = client.get('/health')\n"
            "    assert resp.status_code == 200\n"
            "    assert resp.json['status'] == 'ok'\n"
            "```"
        ),
    },
]
