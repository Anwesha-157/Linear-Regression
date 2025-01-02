from fastapi import FastAPI, HTTPException, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import subprocess
import os

app = FastAPI()

# Dummy user data for login
users = {
    "anwesha_1234": "Ss@19012010"
}

# Mount the 'static' directory to serve the HTML and CSS files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Serve the demo.html file
@app.get("/", response_class=HTMLResponse)
async def serve_login_page():
    with open("static/demo.html", "r") as file:
        html_content = file.read()
    return HTMLResponse(content=html_content, status_code=200)

# Handle login
@app.post("/login")
async def login(username: str = Form(...), password: str = Form(...)):
    if users.get(username) == password:
        # Run scripts after successful login
        try:
            scripts = ["Synthetic.py", "Visualization.py", "Experiment.py", "Cost.py"]
            for script in scripts:
                script_path = os.path.join(os.getcwd(), script)
                result = subprocess.run(["python", script_path], capture_output=True, text=True)
                if result.returncode != 0:
                    return {"message": "Error running scripts", "error": result.stderr}
        except Exception as e:
            return {"message": "Error running scripts", "error": str(e)}

        return {"message": "Login Successful"}
    else:
        raise HTTPException(status_code=401, detail="Invalid Username or Password")
