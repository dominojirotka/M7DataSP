U DenseNN jsem se snažil hrát se všemi paramemetry. Mým osobním cílem bylo dostatat se pře 90% accuracy, ale to se bohužel nepovedlo. Zjistil jsem, že je poměrně efektivní snížit batch na 64 zvýšit počet epoch na 50, což mi přineslo poměrně nízká čísla u loss.

U SumpleCNN jsem měl stejný cíl, a to dostat accuracy přes 90 %, což se mi několikrát povedlo. Konrétně jsem změnšil batch size, přidal jsem vrstvy neuronů a změnil dropout mezi vrstvami na 0.15. Těchto výsledků jsem ale vždy dosáhl s vyšším počtem epoch, konkrétně v nahranémk příkladu se jednalo o 100 epoch.
