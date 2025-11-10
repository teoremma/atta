

def test_attention():
    text_file = "test.txt"
    with open(text_file, "r") as f:
        text = f.read()

    print(text)

if __name__ == "__main__":
    test_attention()