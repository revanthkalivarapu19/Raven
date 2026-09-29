from agents.nlp_pipeline.nlp_pipeline import run_nlp_pipeline


def main():

    text = input("\nEnter news/text: ")

    result = run_nlp_pipeline(text)

    print("\n========== NLP PIPELINE RESULT ==========")
    print("Text:", result["text"])
    print("Tokens:", result["tokens"])
    print("Sentences:", result["sentences"])
    print("POS Tags:", result["pos_tags"])
    print("Entities:", result["entities"])
    print("Dependencies:", result["dependencies"])
    print("Keywords:", result["keywords"])


if __name__ == "__main__":
    main()