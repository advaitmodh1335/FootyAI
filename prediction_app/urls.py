# prediction_app/urls.py
from django.urls import path, re_path
from . import views

urlpatterns = [
    path("health/", views.health, name="health"),
    path("get_teams_and_matches/", views.get_teams_and_matches, name="get_teams_and_matches"),
    path("team_data/", views.team_data, name="team_data"),
    re_path(r"^team_data/(?P<team>.+)/$", views.team_data, name="team_data_with_param"),
    path("predict/", views.predict, name="predict"),
    # Optional legacy alias (only if your frontend or bookmarks still hit /predict_match/)
    path("predict_match/", views.predict_match, name="predict_match"),
]