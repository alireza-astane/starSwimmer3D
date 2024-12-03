#version 3.7; 
    global_settings { assumed_gamma 1.0 }
    

    camera {
    location  <20, 20, 20>
    right     x*image_width/image_height
    look_at   <0, 0, 0>
    angle 58
    }

    background { color rgb<1,1,1>*0.03 }


    light_source { <-20, 30, -25> color red 0.6 green 0.6 blue 0.6 }
    light_source { < 20, 30, -25> color red 0.6 green 0.6 blue 0.6 }
    
    #declare m = 1;
    union {

    union {
    sphere { m*<-0.9841876320067551,-1.0605629106348458,-0.7140564002599432>, 1 }        
    sphere {  m*<0.4472246182692412,-0.23522298126839458,9.14856415947453>, 1 }
    sphere {  m*<7.802576056269218,-0.3241432572627505,-5.430929130570801>, 1 }
    sphere {  m*<-6.217465136919364,5.235427707654818,-3.392833031007869>, 1}
    sphere { m*<-2.2268586923212137,-3.721609172194066,-1.3247494170232965>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.4472246182692412,-0.23522298126839458,9.14856415947453>, <-0.9841876320067551,-1.0605629106348458,-0.7140564002599432>, 0.5 }
    cylinder { m*<7.802576056269218,-0.3241432572627505,-5.430929130570801>, <-0.9841876320067551,-1.0605629106348458,-0.7140564002599432>, 0.5}
    cylinder { m*<-6.217465136919364,5.235427707654818,-3.392833031007869>, <-0.9841876320067551,-1.0605629106348458,-0.7140564002599432>, 0.5 }
    cylinder {  m*<-2.2268586923212137,-3.721609172194066,-1.3247494170232965>, <-0.9841876320067551,-1.0605629106348458,-0.7140564002599432>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    #version 3.7; 
    global_settings { assumed_gamma 1.0 }
    

    camera {
    location  <20, 20, 20>
    right     x*image_width/image_height
    look_at   <0, 0, 0>
    angle 58
    }

    background { color rgb<1,1,1>*0.03 }


    light_source { <-20, 30, -25> color red 0.6 green 0.6 blue 0.6 }
    light_source { < 20, 30, -25> color red 0.6 green 0.6 blue 0.6 }
    
    #declare m = 1;
    union {

    union {
    sphere { m*<-0.9841876320067551,-1.0605629106348458,-0.7140564002599432>, 1 }        
    sphere {  m*<0.4472246182692412,-0.23522298126839458,9.14856415947453>, 1 }
    sphere {  m*<7.802576056269218,-0.3241432572627505,-5.430929130570801>, 1 }
    sphere {  m*<-6.217465136919364,5.235427707654818,-3.392833031007869>, 1}
    sphere { m*<-2.2268586923212137,-3.721609172194066,-1.3247494170232965>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.4472246182692412,-0.23522298126839458,9.14856415947453>, <-0.9841876320067551,-1.0605629106348458,-0.7140564002599432>, 0.5 }
    cylinder { m*<7.802576056269218,-0.3241432572627505,-5.430929130570801>, <-0.9841876320067551,-1.0605629106348458,-0.7140564002599432>, 0.5}
    cylinder { m*<-6.217465136919364,5.235427707654818,-3.392833031007869>, <-0.9841876320067551,-1.0605629106348458,-0.7140564002599432>, 0.5 }
    cylinder {  m*<-2.2268586923212137,-3.721609172194066,-1.3247494170232965>, <-0.9841876320067551,-1.0605629106348458,-0.7140564002599432>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    