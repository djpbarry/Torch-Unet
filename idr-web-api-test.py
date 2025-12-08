import requests

INDEX_PAGE = "https://idr.openmicroscopy.org/webclient/?experimenter=-1"
image_id = 1000650

# create http session
with requests.Session() as session:
    request = requests.Request('GET', INDEX_PAGE)
    prepped = session.prepare_request(request)
    response = session.send(prepped)
    if response.status_code != 200:
        response.raise_for_status()

    qs = {'image_id': image_id}
    IMAGE_DETAILS_URL = "https://idr.openmicroscopy.org/webclient/imgData/{image_id}/"
    url = IMAGE_DETAILS_URL.format(**qs)
    r = session.get(url)
    if r.status_code == 200:
        print(r.json())
