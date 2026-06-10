import requests

def upload_image(image_path):

    url = "http://127.0.0.1:8000/analyze"

    with open('C:/Users/23838/Pictures/Camera Roll/家人/567.jpg', "rb") as f:

        files = {
            "file": f
        }

        response = requests.post(url, files=files)

    print("状态码:", response.status_code)
    print("返回内容:", response.text)