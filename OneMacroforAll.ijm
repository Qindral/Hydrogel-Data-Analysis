macro "Preprocessing Workflow with Log" {

    
    // --- PARAMETER EINSTELLUNGEN ---
    nRandomFrames = 5;
    sigmaRandom = 0; 
    
    windowSize = 4;
    projMethod = "Average Intensity";
    
    smoothRepeats = 5; 
    
    rollingBallRadius = 15;
    
    subFolder = "preprocess";
    // -------------------------------

    if (nImages == 0) {
        showMessage("Fehler", "Bitte öffne zuerst einen Stack und speichere ihn ab.");
        return;
    }

    dir = getDirectory("image");
    name = getTitle();
    dotIndex = lastIndexOf(name, ".");
    if (dotIndex != -1) nameBase = substring(name, 0, dotIndex);
    else nameBase = name;
    
    outputDir = dir + subFolder + File.separator;
    if (!File.exists(outputDir)) File.makeDirectory(outputDir);

    // Logging vorbereiten
    logText = "Macro Execution Log\n";
    getDateAndTime(year, month, dayOfWeek, dayOfMonth, hour, minute, second, msec);
    logText = logText + "Date: " + year + "-" + (month+1) + "-" + dayOfMonth + " " + hour + ":" + minute + "\n";
    logText = logText + "Original File: " + name + "\n";
    logText = logText + "------------------------------------------------\n";

    setBatchMode(true);
    originalID = getImageID();
    
    // Arbeitskopie erstellen
    run("Duplicate...", "title=Processing_Stack duplicate");
    run("32-bit");
    workID = getImageID();

    // -------------------------------------------------------------
    // SCHRITT 1: Random Background
    // -------------------------------------------------------------
    logText = logText + "Step 1: Random Background Subtraction\n";
    logText = logText + "   - NumFrames: " + nRandomFrames + "\n";

    getDimensions(w, h, ch, sl, fr);
    totalFrames = fr; if (sl > fr) totalFrames = sl;
    
    randStackTitle = "Random_Temp_Stack";
    selectImage(originalID);
    r1 = floor(random() * totalFrames) + 1;
    run("Duplicate...", "duplicate range="+r1+"-"+r1);
    rename(randStackTitle);
    
    for (k = 1; k < nRandomFrames; k++) {
        selectImage(originalID);
        rn = floor(random() * totalFrames) + 1;
        run("Duplicate...", "duplicate range="+rn+"-"+rn);
        rename("Temp_Slice");
        run("Concatenate...", "  title=["+randStackTitle+"] image1=["+randStackTitle+"] image2=[Temp_Slice]");
    }
    
    selectImage(randStackTitle);
    run("Z Project...", "projection=[Average Intensity]");
    bgImageID = getImageID();
    rename("Calculated_Background");
    selectImage(randStackTitle); close();
    
    imageCalculator("Subtract stack", "Processing_Stack", "Calculated_Background");
    selectImage(bgImageID); close();

// SCHRITT 2: Sliding Window (STRICT - Keine Ränder)
    // -------------------------------------------------------------
    logText = logText + "Step 2: Sliding Window Average (Strict)\n";
    logText = logText + "   - Window Size: " + windowSize + "\n";
    
    // Berechne, wie viele "gültige" Frames wir haben
    // Wenn wir 4 Frames brauchen, können wir bei 300 nur bis 297 gehen.
    validFrames = totalFrames - windowSize + 1;
    
    if (validFrames < 1) {
        exit("Fehler: Das Video ist kürzer als die Fenstergröße!");
    }
    
    logText = logText + "   - Output Frames: " + validFrames + " (cut off end)\n";

    newImage("Sliding_Result", "32-bit black", w, h, validFrames);
    resultSlidingID = getImageID();
    
    selectImage(workID);
    
    // Schleife läuft nur bis 'validFrames'
    for (i = 1; i <= validFrames; i++) {
        
        startF = i;
        endF = i + windowSize - 1;
        // Kein "if endF > totalFrames" mehr nötig, da die Schleife vorher stoppt.
        
        selectImage(workID);
        run("Duplicate...", "duplicate range=" + startF + "-" + endF);
        tempWinID = getImageID();
        
        // Da wir sicher sind, dass wir 'windowSize' Bilder haben,
        // können wir direkt projizieren ohne Edge-Case-Checks.
        run("Z Project...", "projection=["+projMethod+"]");
        tempProjID = getImageID();
        
        // Kopieren und Einfügen
        selectImage(tempProjID); 
        run("Copy");
        
        selectImage(resultSlidingID); 
        setSlice(i); 
        run("Paste");
        
        // Aufräumen
        selectImage(tempWinID); close();
        selectImage(tempProjID); close();
    }
    
    // Den alten (langen) Stack schließen und durch den neuen (kurzen) ersetzen
    selectImage(workID); close();
    selectImage(resultSlidingID);
    rename("Processing_Stack");
    workID = getImageID();

    // -------------------------------------------------------------
    // SCHRITT 3: Smoothing
    // -------------------------------------------------------------
    logText = logText + "Step 3: Smoothing (3x)\n";
    selectImage(workID);
    for (s = 0; s < smoothRepeats; s++) {
        run("Smooth", "stack");
    }


    // -------------------------------------------------------------
    // SCHRITT 4: Final Background
    // -------------------------------------------------------------
    logText = logText + "Step 4: Subtract Background (Rolling Ball 15px)\n";
    run("Subtract Background...", "rolling="+rollingBallRadius+" disable stack");


    // -------------------------------------------------------------
    // SPEICHERN
    // -------------------------------------------------------------
    finalName = nameBase + "_processed.tif";
    savePath = outputDir + finalName;
    saveAs("Tiff", savePath);
    
    logPath = outputDir + nameBase + "_log.txt";
    File.saveString(logText, logPath);
    
    setBatchMode(false);
   // showMessage("Fertig", "Gespeichert in: " + outputDir);
}