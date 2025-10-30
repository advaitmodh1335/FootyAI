from django.db import models

# Create your models here.
from django.db import models

class Match(models.Model):
    # Match details
    date = models.DateField()
    time = models.TimeField()
    comp = models.CharField(max_length=50)
    round = models.CharField(max_length=50)
    day = models.CharField(max_length=10)
    venue = models.CharField(max_length=50)
    team = models.CharField(max_length=50)
    opponent = models.CharField(max_length=50)
    result = models.CharField(max_length=1)
    gf = models.FloatField()
    ga = models.FloatField()
    sh = models.FloatField()
    sot = models.FloatField()
    xg = models.FloatField()
    xga = models.FloatField()
    poss = models.FloatField()
    season = models.CharField(max_length=10)
    captain = models.CharField(max_length=50)
    formation = models.CharField(max_length=50)
    opp_formation = models.CharField(max_length=50)
    referee = models.CharField(max_length=50)

    # Encoded features for the model
    team_code = models.IntegerField()
    opp_code = models.IntegerField()
    venue_code = models.IntegerField()
    target = models.IntegerField()

    def __str__(self):
        return f"{self.team} vs {self.opponent} on {self.date}"
