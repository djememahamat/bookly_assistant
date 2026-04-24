from dotenv import load_dotenv
import os

load_dotenv()

def main():
    print("Decagon Interview - Case Study")
    print(os.getenv("LLM_MODEL"))


if __name__ == "__main__":
    main()
