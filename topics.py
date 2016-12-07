import math

from nltk.corpus import stopwords
from textblob import TextBlob as tb


def tf(word, blob):
    return blob.words.count(word) / len(blob.words)


def n_containing(word, bloblist):
    return sum(1 for blob in bloblist if word in blob.words)


def idf(word, bloblist):
    return math.log(len(bloblist) / (1 + n_containing(word, bloblist)))


def tfidf(word, blob, bloblist):
    return tf(word, blob) * idf(word, bloblist)


document1 = tb("""Rovviltmeldingen la ikke opp til noen radikal endring av rovviltforvaltningen,
selv om det ble lagt større vekt på en bestandsrettet forvaltning enn tidligere. Stortingets behandling
av meldingen innebar en stadfesting av hovedprinsippene i dagjeldende forvaltning, samt at
Stortinget gav retningslinjer for den framtidige utøvelse av forvaltningen.Som en oppfølging av
rovviltmeldingen fastsatte Miljøverndepartementet 8. mars i år nye erstatningsregler for rovviltskader
på husdyr. Reglene bygger i all hovedsak på tidligere regelverk på området, men avgjørelsesmyndigheten
i erstatningssaker er delegert fra Direktoratet for naturforvaltning til fylkesmennene. Dette innebærer
en effektivisering av erstatningsoppgjøret. Selv om innføringen av nye rutiner i år innebar at skadeoppgjøret
for beitesesongen 1992 ble noe forsinket, vil de nye reglene medføre at utbetalingene heretter vil skje
vesentlig raskere enn hva som har vært vanlig tidligere. For skadeåret 1993 tas det sikte på at utbetaling
vil skje i januar 1994.""")

document2 = tb("""Jeg tillater meg å stille følgende spørsmål til samferdselsministeren:Hovedplanen for parsellen Håland  Teigland langs Åkrafjorden  Rv 11  er anket inn for avgjørelse i Samferdselsdepartementet. Dette fører til at det bompengefinansierte vegprosjektet ikke kan fullføres innen 1995 som forutsatt.Når vil departementet ta stilling til anken over Vegdirektoratets standpunkt til trasévalg?
""")

document3 = tb("""Drapsforsøket mot forlagssjef William Nygaard var svært alvorlig, og Oslo politikammer vil sette svært mye inn på å oppklare den saken.Uavhengig av denne saken er Overvåkingstjenesten opptatt av som et ledd i å bekjempe terrorisme å ha oppmerksomhet rettet mot aktuelle utenlandske miljøer etter de retningslinjer som blant annet er omtalt i overvåkingsmeldingen, som ble behandlet i Stortinget nå i vår.Det har ikke vært norsk praksis å drive utstrakt kartlegging av alle etniske og religiøse miljøer og grupperinger her i landet. Regjeringen ønsker en mer offensiv kamp mot kriminaliteten, også den organiserte kriminaliteten, og det er naturlig at vi i dette helhetlige perspektivet også har oppmerksomheten rettet inn mot en bedre forebyggelse av terrorisme.
""")


stop = set(stopwords.words('norwegian'))
print(stop)
bloblist = [document1, document2, document3]
for i, blob in enumerate(bloblist):
    print("Top words in document {}".format(i + 1))
    scores = {word: tfidf(word, blob, bloblist) for word in blob.words if word not in stop}
    sorted_words = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    for word, score in sorted_words[:3]:
        print("\tWord: {}, TF-IDF: {}".format(word, round(score, 5)))
