from fasthtml.common import *
from pathlib import Path

app, rt = fast_app(debug=True)

upload_dir = Path("filez")
upload_dir.mkdir(exist_ok=True)

@rt('/')
def get():
    return Titled("File Upload Demo",
        Article(
            Form(hx_post=upload, hx_target="#result-one")(
                Input(type="file", name="file"),
                Button("Upload", type="submit", cls='secondary'),
            ),
            Div(id="result-one")
        )
    )

def FileMetaDataCard(file):
    print(file.filename)
    return Article(
        Header(H3(file.filename)),
        Ul(
            Li('Size: ', file.size),            
            Li('Content Type: ', file.content_type),
            Li('Headers: ', file.headers),
        )
    )    

@rt
async def upload(file: UploadFile):
    print(f"receiving file {file.filename}")
    card = FileMetaDataCard(file)
    filebuffer = await file.read()
    print("store file")
    (upload_dir / file.filename).write_bytes(filebuffer)
    return card

serve()
