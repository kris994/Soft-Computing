from userimageski import UserData

if __name__ == '__main__':

    user = UserData('test02.jpeg') #Ucitava sliku
    user.plot_preprocessed_image() #Slika se procesira
    candidates = user.get_text_candidates() #Detektuju se objekti u slici
    user.plot_to_check(candidates, 'Total Objects Detected') #Izdvoji pronadjene objekte
    maybe_text = user.select_text_among_candidates('C:\Users\krist\Desktop\PrepoznavanjeSlovaNaSlici\linearsvc-hog-fulltrain2-90.pickle') #Detktuje se tekst izedju prethodno pronadjenih objekata
    user.plot_to_check(maybe_text, 'Objects Containing Text Detected') #Izdvoji objekte sa tekstom
    classified = user.classify_text('C:\Users\krist\Desktop\PrepoznavanjeSlovaNaSlici\linearsvc-hog-fulltrain36-90.pickle') #Prepoznaj slovo
    user.plot_to_check(classified, 'Single Character Recognition') #Ispise slova nakon detekcije ispred odgovarajuceg kvadrata
    user.realign_text() #Ispise slova na mestu gde bi bili kod slike