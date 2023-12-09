from fastapi import FastAPI, HTTPException, File, UploadFile
from routes.route import router


app = FastAPI()

app.include_router(router)










