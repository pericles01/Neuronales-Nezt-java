import java.util.Random;

/**
 * Diese Klasse repräsentiert ein neuronales Netz
 * 
 *    ---  
 *    ---Summe u --> Neuron mit sigmoider Funktion ---> out
 *    ---
 *    
 *    Gewichte:
 *    w[p][q][r]
 *    p legt fest, in zwischen welchen Neuronenschichten die Gew. liegen.
 *    q entspricht der Anzahl der Eingänge der Nachfolgeschicht
 *    r entspricht der Anzahl der Ausgänge der Vorgängerschicht
 *    
 *    Beispiel für p festgelegt z.B. p=2
 *    
 *    w00 w01 ... w0[r-1]
 *    w10 w11 ... w1[r-]
 *    ...
 *    w[q-1]0 w[q-1]1 ... w[q-1][r-1]
 *    
 *    ... und alle u-Werte der Nachfolge schicht ergeben sich dann
 *    als:  u = w[p] * out
 *    (out-Vektor der vorangehenden Schicht)
 *    
 *    ----------------------------------------------
 *    
 *    Wenn es m Neuronenschichten gibt, dann gibt es m-1 zweidimensionale Arrays von w, also w[0..m-2][..][..]
 *    
 *    Beispiel:
 *    Neuronenschicht-Index:      0    1   2   3
 *    w-Index (erste Dimension):    0    1   2
 */
public class NeuroNetwork {
	
	private double d;
	static double[][][] w;
	private double [][] out;
	private double [][] u;
	private int anzahlNeuroSchichten;
	private int anzahlVerbindungsSchichten;
	
	//Für Backpropagation:
    //  x ... immer soviel wie vorangehende Schicht Ausgänge hat.
    //  Q ... immer soviel wie Nachfolgeschicht (z.B. Ausgangslayer) Neuronen hat.
    // jedoch hier:
    // jedem w ein x und ein Q zuordnen, um die Handhabung zu vereinfachen!
	private double[][][] x;
	private double[][][] Q;

    /**
     *    Dem Konstruktor wird die Anazhl der Eingangsneuronen übergeben: anzEingang (Skalar)<br/>
     *    Dem Konstruktor wird die Anazhl der Neuronen jeder verdenkten Schicht inklusive der Ausgangsschicht übergeben: int[] anzMitte (Array)<br/>
    */   
	public NeuroNetwork(int anzEingang, int[] anzMitte, double d) {
		this.d=d;
		this.anzahlNeuroSchichten=1+anzMitte.length;
		this.anzahlVerbindungsSchichten=this.anzahlNeuroSchichten-1;
		
		w=new double[anzahlVerbindungsSchichten][][];
		x=new double[anzahlVerbindungsSchichten][][];
		Q=new double[anzahlVerbindungsSchichten][][];
		
		w[0]= new double[anzMitte[0]][anzEingang];
		x[0]= new double[anzMitte[0]][anzEingang];
		Q[0]= new double[anzMitte[0]][anzEingang];
		
		for(int i=1;i<w.length;i++)
        {
             w[i] = new double[ anzMitte[i] ][ anzMitte[i-1] ];
             x[i] = new double[ anzMitte[i] ][ anzMitte[i-1] ];
             Q[i] = new double[ anzMitte[i] ][ anzMitte[i-1] ];
        }
        
        out = new double[anzahlNeuroSchichten][];  //Ausgang eines Neurons, also sigmoid(u)
        u   = new double[anzahlNeuroSchichten][];  //gewichtete Summe am Eingang
        
        out[0] = new double[anzEingang];
        u[0]   = new double[anzEingang];
        
        for(int i=0;i<anzMitte.length;i++)
        {
              out[i+1] = new double[anzMitte[i]];
              u[i+1] = new double[anzMitte[i]];
        }
    }
	
	private double sigmoid(double u) {
		return 1/(1+Math.exp(-d*u));
	}
	
	private void sigmoid(double[] out,double[] u) {
		for(int i=0;i<u.length;i++)
			out[i]=sigmoid(u[i]);
	}
	/**
     *   @param ergebnisvektor liefert Ergebnis der Matrizenmultiplikation ergebnisvektor = matrix * input
     *   <pre>
     *   Beispiel: Matrix ist 3x2 Matrix, input:2x1, ergebnis:3x1
     *   
     *   ergebnisvektor0 = matrix00*input0 + matrix01*input1
     *   ergebnisvektor1 = matrix10*input0 + matrix11*input1
     *   ergebnisvektor2 = matrix20*input0 + matrix21*input1
     */
	private void matrixMutiplikation(double[] ergebnis, double[][]wMatrix,double[] inputMatrix) {
		for(int zeile=0;zeile<ergebnis.length;zeile++) {
			ergebnis[zeile]=0;
			for(int spalte=0;spalte<inputMatrix.length;spalte++) {
				ergebnis[zeile]+=wMatrix[zeile][spalte]*inputMatrix[spalte];
			}
		}
	}
	
	private void vorwaertsPropagieren(int vonSchicht, int nachSchicht) {
		double[][] ww=w[vonSchicht];
		matrixMutiplikation(u[nachSchicht],ww,out[vonSchicht]);
		sigmoid(out[nachSchicht],u[nachSchicht]);
	}
	/**
     * Dieser Methode wird als Referenz die Eingänge übergeben und in die übergebene ausgangs-Referenz
     * wird das Ergebnis der Vorwärtspropagierung geschrieben, also die Ausgangswerte der letzten Schicht.
     */
    public boolean vorwaertsPropagieren(double[] ausgang, double[] eingang)
    {
        if(ausgang==null || ausgang.length!=out[out.length-1].length)
        {
             System.out.println("Übergebene Referenz ausgang muss die Länge "+out[out.length-1].length+" haben!");
             return false;
        }
        if(eingang==null || eingang.length!=out[0].length)
        {
             System.out.println("Übergebene Referenz eingang muss die Länge "+out[0].length+" haben!");
             return false;
        }
        
        //1. Kopierren des von aussen übergebenen Eingangsvektors auf u und out der ersten "Neuronenschicht"
        //   Da das die Eingangsschricht ist, wird dort nicht die sigmoide Funktion zwischen u und out angewendet,
        //   sondern u == out
        for(int i=0; i<u[0].length;i++)
        {
               u[0][i] = eingang[i];
             out[0][i] = eingang[i];
        }
        
        //2. Schicht für Schicht die Signale durchpropagieren mit u_nach = w_aktuell * out_vor 
        //   und sigmoid(out_nach)
        //  ...
        for(int i=0;i<out.length-1;i++)
        {
            vorwaertsPropagieren(i,i+1);
        }
        
        //3. Übertragen der Ausgänge der letzten Schicht auf die von aussen übergebene Referenz ausgang:
        for(int i=0;i<out[out.length-1].length;i++)
        {
             ausgang[i] = out[out.length-1][i];
        }
        
        return true;
    }
    
    //------------------------------------------------
    //---------- Backpropagation ---------------------
    //------------------------------------------------
    Random zufall=new Random(System.currentTimeMillis());
    
    private double ableitung_sigmoid(double u) {
    	return d*Math.exp(-d*u)/((1+Math.exp(-d*u))*(1+Math.exp(-d*u)));
    }
    private void ableitung_sigmoid(double[] out, double[] u) {
    	for(int i=0;i<u.length;i++)
    		out[i]=ableitung_sigmoid(u[i]);
    }
    /**
     *   allen w Zufallszahlen in ]-RANGE,+RANGE[ zuweisen
     */
    public void initGewichteZufaellig(double RANGE)
    {
           for(int i=0;i<w.length;i++)
               for(int k=0;k<w[i].length;k++)
                   for(int p=0;p<w[i][k].length;p++)
                       w[i][k][p] = RANGE*2.0*(zufall.nextDouble()-0.5);
    }
    
    /**
     * <pre>
     *   Backpropagation an der Ausgangsschicht:
     *   w_neu = w_alt + lernfaktor * x * Q 
     *   
     *   Q = f'(u)*(out_soll-out_ist)
     *   
     *   f'(u): Ableitung der (stetigen) sigmoiden Funktion nach dem Eingang u.
     *   x: Aktivierung (Eingangssignal) der Leitung dessen Gewicht gerade betrachtet wird.
     *
     *   Zwischenschichten:
     *   w_neu = w_alt + lernfaktor*x*Q
     *   
     *   Q = f'(u)* SUMME Qk * wneuk
     *   
     *   Qk: Q-Werte Aller vom Zielneuron abgehenden Gewichtsleitungen (Nachfolgeschicht). Diese sind bekannt, da Backpropagation am Ausgang beginnt und dann Schicht für Schicht zum Eingang hin weitergeführt wird.
     *   wneu k: Neu berechnete Gewichtswerte aller vom Zielneuron abgehenden Gewichtsleitungen.
     *
     *
     * </pre>
     * 
     * Führt EINEN Backpropagationschritt durch und liefert 
     * den aktuellen Gesamtfehler.
     */        
    public void backpropagationSchritt(double[] eingang, double[] soll_ausgang, double lernfaktor)
    {
        // 1. Vorwärtspropagieren mit dem gegebenen Eingang
        vorwaertsPropagieren(out[out.length-1],eingang);
        
        // 2. Backpropagation für die Ausgangsschicht
        // 2a) Bestimmung der Q-Werte
        for(int i=0;i<out[out.length-1].length;i++)  // Anzahl der Zeilen der entsprechenden w-Matrix
        {
               double uu = u[out.length-1][i];
               double differenz = soll_ausgang[i] - out[out.length-1][i];
               double QQ = ableitung_sigmoid(uu)*differenz;
               
               for(int k=0;k<out[out.length-2].length;k++) // Anzahl der Spalten der entsprechenden w-Matrix
               {
                     Q[out.length-2][i][k] = QQ;
               }
        }
        // 2b) Bestimmung der x-Werte 
        for(int k=0;k<out[out.length-2].length;k++)
        {
               double xx = out[out.length-2][k];
               for(int i=0;i<out[out.length-1].length;i++)
               {
                     x[out.length-2][i][k] = xx;
               }
        }
        
        // 2c) Modifikation der Gewichte der letzten Verbindungsschicht:
        for(int i=0;i<out[out.length-1].length;i++)
        {
              for(int k=0;k<out[out.length-2].length;k++)
              {
                    w[out.length-2][i][k] += lernfaktor * x[out.length-2][i][k] * Q[out.length-2][i][k];
              }
        }
        
        // 3. Backpropagation für alle Zwischenschichten rückwärts
        for(int WSCHICHT=out.length-3;WSCHICHT>=0;WSCHICHT--)
        {
            // 3a) Bestimmung der Q-Werte
            for(int i=0;i<out[WSCHICHT+1].length;i++)  // Anzahl der Zeilen der entsprechenden w-Matrix
            {
               double uu = u[WSCHICHT+1][i];
               //double differenz = soll_ausgang[i] - out[out.length-1][i];
               //double QQ = ableitung_sigmoid(uu)*differenz;
               
               //neue Gewichte mal Q-Werte zu den Leitungen, die von dem aktuellen Neuron abgehen.
               //aktuelles Neuron liegt rechts in der aktuellen Verbindungsschicht:
               double SUMME = 0.0;
               for(int p=0;p<out[WSCHICHT+2].length;p++)
               {
                    SUMME += w[WSCHICHT+1][p][i]*Q[WSCHICHT+1][p][i];
               }
               double QQ = ableitung_sigmoid(uu)*SUMME;
               for(int k=0;k<out[WSCHICHT].length;k++) // Anzahl der Spalten der entsprechenden w-Matrix
               {
                     Q[WSCHICHT][i][k] = QQ;
               }
            } 
            // 3b) Bestimmung der x-Werte 
            for(int k=0;k<out[WSCHICHT].length;k++)
            {
               double xx = out[WSCHICHT][k];
               for(int i=0;i<out[WSCHICHT+1].length;i++)
               {
                     x[WSCHICHT][i][k] = xx;
               }
            }   
            // 3c) Modifikation der Gewichte der letzten Verbindungsschicht:
            for(int i=0;i<out[WSCHICHT+1].length;i++)
            {
              for(int k=0;k<out[WSCHICHT].length;k++)
              {
                    w[WSCHICHT][i][k] += lernfaktor * x[WSCHICHT][i][k] * Q[WSCHICHT][i][k];
              }
            }            
        }    
    }    
    
    public double berechneAktuellenFehler(double[] eingang, double[] soll_ausgang)
    {
          vorwaertsPropagieren(out[out.length-1],eingang);
          
          double FEHLER = 0.0;
          for(int i=0;i<out[out.length-1].length;i++)
              FEHLER += Math.abs( soll_ausgang[i] - out[out.length-1][i] );
              
          return FEHLER;    
    }
}

