import os


class ModelRetriever(object):

    @classmethod
    def get_model_filepath(cls, model_name, model_directory="models"):

        if model_name == "bedrooms":
            return os.path.join(model_directory, "karras2019stylegan-bedrooms-256x256.pkl")

        elif model_name == "cars":
            return os.path.join(model_directory, "karras2019stylegan-cars-512x384.pkl")

        elif model_name == "cats":
            return os.path.join(model_directory, "karras2019stylegan-cats-256x256.pkl")

        elif model_name == "celebahq":
            return os.path.join(model_directory, "karras2019stylegan-celebahq-1024x1024.pkl")

        elif model_name == "ffhq":
            return os.path.join(model_directory, "karras2019stylegan-ffhq-1024x1024.pkl")

        raise Exception(model_name + " model unknown")
