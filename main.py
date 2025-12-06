def main():
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    print(os.getenv("GLM_4_FLASH_API_KRY"))


if __name__ == "__main__":
    main()
