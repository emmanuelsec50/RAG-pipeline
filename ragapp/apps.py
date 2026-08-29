from django.apps import AppConfig


# class RagappConfig(AppConfig):
#     name = 'ragapp'




class RagappConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'ragapp'

    def ready(self):
        # Deferred import: importing knowledge_base at module load time
        # (rather than inside ready()) risks running before Django's app
        # registry is fully populated, and more practically, means the
        # (slow) torch.load call would happen at import time rather than
        # at the deliberate, single "app starting up" moment.
        from . import knowledge_base
        knowledge_base.load()