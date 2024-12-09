Innholdi mappene er som følger:

Rapport og vedlegg inneholder den endelige rapporten fra prosjektet samt vedlegg som det henvises til i rapporten

Benyttet programkode samt plott og data som er lagret fra kjøringene finnes i mappene Testkjøringer og Endelige_kjøringer
Fordi jeg har kjørt programmene fra min google drive og lagret data der underveis, antar jeg (håper jeg!) det er nødvendig å kommentere ut de delene av koden som har med kobling til google drive, lagring av plott og skriving av tekstfiler for at de skal kunne kjøre.


Mappen Testkjøringer inneholder undermappene FFNN og CNN som inneholder programmer og resultater fra utprøvingene som er gjort med MNIST-data.

	FFNN inneholder jupyter notebook-filer for testkjøringer på MNIST med logistisk regresjon og FFNN med ett og to skjulte lag. For de nevrale nettene er det lagt inn flere ulike modeller. Man må kommentere ut de modellene som ikke skal benyttes, slik at det kun står en modell igjen nå koden kjøres.

	CNN inneholder jupyter notebook-filer for kjøringer av 
		konvolusjonsnettverk med ett og to konvolusjonslag, 
		en egen fil for kjøring med stride=2 fordi det påvirker dimensjonaliteten
		egne filer for ett og to konvolusjonslag med fullt koblede skjulte lag på toppen
		en fil for LeNet5-arkitekturen
	
	Undermappene 1lag,2lag og LeNet5 inneholder data og plott fra kjøringene med henholsdvis ett og to konvolusjonslag og LeNet5-arkitekturen.
	
	For de nevrale nettene er det lagt inn flere ulike modeller. Man må kommentere ut de modellene som ikke skal benyttes, slik at det kun står en modell igjen nå koden kjøres.



Mappen Endelige_kjøringer inneholder det som ble funnet som de beste modellene og jupyter notebook-filer for å kjøre disse med både MNIST og Fashion MNIST. I tillegg finnes plot og data lagret fra disse kjøringene der.
Jupyter notebookfilen plot plotter resultatene fra alle kjøringene sammen.