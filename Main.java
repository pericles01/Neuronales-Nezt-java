import java.util.Random;

public class Main {
	
	
	public static void main(String[] args) {
		Random zufall = new Random(System.currentTimeMillis());
		double ddd = 1.4;
		double[][][] www = {
		  {
		      {-1.1131178942900224,0.2879801654988282},
		      {13.736138215249058,-12.609846580263223}
		  },
		  {
		      {-0.8151251394319519,-3.204746485543899},
		      {-0.6978323986439555,-3.1384961491201264}
		  },
		  {
		      {4.255852071707854,-3.800630854952885}
		  },
		                 };

	     NeuroNetwork netz = new NeuroNetwork(2, new int[] {2,2,1},ddd);
	     netz.w= www;
	     
	     double[] output = new double[1];
	     
	     //  in0=-1.0 in1=-1.0 soll=0.0 ist:0.49919912471348593
	     //  vorwaertsPropagieren(double[] ausgang, double[] eingang)
	     netz.vorwaertsPropagieren(output, new double[] {-1.0,-1.0});
	     System.out.println("in: -1 -1 out:"+output[0]+" in Testprogramm vorher: 0.49919912471348593");
	     //  in0=-1.0 in1=1.0 soll=1.0 ist:0.5039651950719938
	     netz.vorwaertsPropagieren(output, new double[] {-1.0,1.0});
	     System.out.println("in: -1 1 out:"+output[0]+" in Testprogramm vorher: 0.5039651950719938");
	     //  in0=1.0 in1=-1.0 soll=1.0 ist:0.5000185751153124
	     netz.vorwaertsPropagieren(output, new double[] {1.0,-1.0});
	     System.out.println("in: 1 -1 out:"+output[0]+" in Testprogramm vorher: 0.5000185751153124");
	     //  in0=1.0 in1=1.0 soll=0.0 ist:0.4999772380555142
	     netz.vorwaertsPropagieren(output, new double[] {1.0,1.0});
	     System.out.println("in: 1 1 out:"+output[0]+" in Testprogramm vorher: 0.4999772380555142");
	     
	     /*
	     LIEFERT (Konsolenausgabe):
	     in: -1 -1 out:0.49919913641136243 in Testprogramm vorher: 0.49919912471348593
	     in: -1 1 out:0.5039652060903305 in Testprogramm vorher: 0.5039651950719938
	     in: 1 -1 out:0.5000185760112115 in Testprogramm vorher: 0.5000185751153124
	     in: 1 1 out:0.4999772392646055 in Testprogramm vorher: 0.4999772380555142     
	     */
	     
	     System.out.println("################# BACKPROPAGATION ################################");
	     netz = new NeuroNetwork(2, new int[] {6,6,1},ddd);
	     
	     System.out.println(" 1. Zufallsinitialisierung der Gewichte ");
	     double RANGE = 2.0;
	     netz.initGewichteZufaellig(RANGE);
	     
	     System.out.println(" 2. Patterns und Soll-Ausgaben zur Verfügung stellen: ");
	     
	     //{ in0 in1 } {Out-soll}:
	     double[][][] pattern = {       
	       {  {0.0,0.0}, {0.0}  },
	       {  {0.0, 1.0}, {1.0}  },
	       {  { 1.0,0.0}, {1.0}  },
	       {  { 1.0, 1.0}, {0.0}  }
	     };
	     
	     System.out.println(" 3. Durchschnittlichen Absolutfehler zu Beginn bestimmen: ");
	     
	     double FEHLER = (  netz.berechneAktuellenFehler(pattern[0][0], pattern[0][1])
	                       +netz.berechneAktuellenFehler(pattern[1][0], pattern[1][1])
	                       +netz.berechneAktuellenFehler(pattern[2][0], pattern[2][1])
	                       +netz.berechneAktuellenFehler(pattern[3][0], pattern[3][1]) )/4.0;
	                       
	     System.out.println("FEHLER = "+FEHLER);
	     
	     
	     double LERNFAKTOR = 0.6;
	     System.out.println("4. Lernschritte durchführen, lernfaktor="+LERNFAKTOR+" d="+ddd);
	     
	     for(int i=0;i<4000;i++)
	     {
	          int INDEX = zufall.nextInt(pattern.length);
	          netz.backpropagationSchritt(pattern[INDEX][0], pattern[INDEX][1], LERNFAKTOR);
	          
	          if(i>0 && i%10==0)
	          {
	              FEHLER = (  netz.berechneAktuellenFehler(pattern[0][0], pattern[0][1])
	                       +netz.berechneAktuellenFehler(pattern[1][0], pattern[1][1])
	                       +netz.berechneAktuellenFehler(pattern[2][0], pattern[2][1])
	                       +netz.berechneAktuellenFehler(pattern[3][0], pattern[3][1]) )/4.0;
	                       
	              System.out.println("FEHLER nach "+i+" Schritten = "+FEHLER);
	          }
	     }

	     FEHLER = (  netz.berechneAktuellenFehler(pattern[0][0], pattern[0][1])
	                       +netz.berechneAktuellenFehler(pattern[1][0], pattern[1][1])
	                       +netz.berechneAktuellenFehler(pattern[2][0], pattern[2][1])
	                       +netz.berechneAktuellenFehler(pattern[3][0], pattern[3][1]) )/4.0;
	                       
	     System.out.println("FEHLER am Ende = "+FEHLER);
	        
	     System.out.println("TEST:");
	     
	     for(int i=0;i<pattern.length;i++)
	     {
	           netz.vorwaertsPropagieren(output, pattern[i][0]);
	           System.out.println("in0="+pattern[i][0][0]+" in1="+pattern[i][0][1]+" ist_out="+output[0]+" soll:"+pattern[i][1][0]);
	     }

	}

}
