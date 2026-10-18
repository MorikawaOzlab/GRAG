"""Map editor. Run: python -m gridworld.map_app (port 5001)."""

from urllib.parse import unquote

from gridworld.storage import list_maps, map_path, read_json, save_map, validate_map
from gridworld.web_common import Handler, serve


class MapHandler(Handler):
    page = "map.html"

    def get_api(self, path):
        if path == "/api/maps":
            return self.json(list_maps())
        if path.startswith("/api/maps/"):
            tail = path[len("/api/maps/"):]
            download = tail.endswith("/download")
            name = unquote(tail[:-len("/download")] if download else tail)
            file_path = map_path(name)
            if download:
                return self.file(file_path, "application/json", "map.json")
            return self.json(validate_map(read_json(file_path)))
        return super().get_api(path)

    def post_api(self, path, body):
        if path == "/api/maps":
            return self.json(save_map(self.parse_json(body)))
        if path == "/api/maps/upload":
            return self.json(save_map(self.upload_json(body)))
        return super().post_api(path, body)


if __name__ == "__main__":
    serve(MapHandler, 5001)
