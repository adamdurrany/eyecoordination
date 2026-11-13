import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "eyecomm.server.api:app",
        host="0.0.0.0",
        port=8765,
        reload=False
    )
