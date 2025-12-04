from modal import App, Image

app = App("hello-world")

@app.function()
def hello():
    print("Hello from Modal!")
    return "Hello, world!"

# This is the critical part - you must use this decorator
@app.local_entrypoint()
def main():
    print("Starting main...")
    result = hello.remote()
    print(f"Got result: {result}")

# Do NOT call main() directly
# Modal handles this for you when you run the script