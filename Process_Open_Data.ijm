macro "Run Specific Macro on All Open Images" {

    if (nImages == 0) {
        showMessage("Fehler", "Keine Bilder geöffnet.");
        return;
    }

    // 1. Frage den User: Welches Makro soll ausgeführt werden?
    // Es öffnet sich ein Datei-Auswahl-Fenster.
    macroPath = File.openDialog("Wähle dein Analyse-Makro (.ijm) aus");
    
    // Abbruch, falls nichts gewählt wurde
    if (lengthOf(macroPath) == 0) return;

    print("\\Clear");
    print("Starte Batch-Run mit Makro: " + File.getName(macroPath));
    print("---------------------------------------------------");

    // 2. IDs aller offenen Bilder sammeln
    // Das ist wichtig, damit wir nicht versehentlich die *Ergebnisse* analysieren, 
    // die das Makro zwischendurch erstellt.
    setBatchMode(true);
    
    count = nImages;
    imageIDs = newArray(count);
    
    for (i = 0; i < count; i++) {
        selectImage(i + 1);
        imageIDs[i] = getImageID();
    }

    // 3. Schleife durch alle gesammelten IDs
    for (i = 0; i < count; i++) {
        
        currentID = imageIDs[i];
        
        // Sicherheitscheck: Ist das Bild noch offen?
        if (!isOpen(currentID)) continue;
        
        selectImage(currentID);
        title = getTitle();
        
        // WICHTIG: Das Bild muss gespeichert sein, damit dein Analyse-Makro
        // weiß, wo es den "preprocess" Ordner erstellen soll.
        dir = getInfo("image.directory");
        if (dir == "") {
            print("ÜBERSPRUNGEN: " + title + " (Nicht gespeichert / kein Pfad)");
            continue;
        }

        print(">> Bearbeite Bild " + (i+1) + " von " + count + ": " + title);
        
        // --- HIER RUFEN WIR DEIN ANDERES MAKRO AUF ---
        // runMacro übergibt die Kontrolle an die .ijm Datei.
        // Das Analyse-Makro arbeitet auf dem aktuell ausgewählten Bild (currentID).
        runMacro(macroPath);
        
        // Nach der Rückkehr stellen wir sicher, dass wir sauber weitermachen.
        // Dein Analyse-Makro schließt am Ende das Resultat (hoffentlich) 
        // oder lässt es offen. Wir wählen für den nächsten Durchlauf nichts aus,
        // das macht der Loop am Anfang.
    }

    setBatchMode(false);
    showMessage("Batch Fertig", "Das gewählte Makro wurde auf alle " + count + " Bilder angewendet.");
}