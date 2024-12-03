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
    sphere { m*<-1.0220397207861596,-1.0139449825183713,-0.7334384808710246>, 1 }        
    sphere {  m*<0.41136934199607667,-0.2055059622484244,9.1302927947401>, 1 }
    sphere {  m*<7.766720779996052,-0.29442623824278064,-5.449200495305236>, 1 }
    sphere {  m*<-6.056375372319778,5.078760804812143,-3.310602362923289>, 1}
    sphere { m*<-2.2717144446562285,-3.6708888287680965,-1.3476900676460621>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.41136934199607667,-0.2055059622484244,9.1302927947401>, <-1.0220397207861596,-1.0139449825183713,-0.7334384808710246>, 0.5 }
    cylinder { m*<7.766720779996052,-0.29442623824278064,-5.449200495305236>, <-1.0220397207861596,-1.0139449825183713,-0.7334384808710246>, 0.5}
    cylinder { m*<-6.056375372319778,5.078760804812143,-3.310602362923289>, <-1.0220397207861596,-1.0139449825183713,-0.7334384808710246>, 0.5 }
    cylinder {  m*<-2.2717144446562285,-3.6708888287680965,-1.3476900676460621>, <-1.0220397207861596,-1.0139449825183713,-0.7334384808710246>, 0.5}

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
    sphere { m*<-1.0220397207861596,-1.0139449825183713,-0.7334384808710246>, 1 }        
    sphere {  m*<0.41136934199607667,-0.2055059622484244,9.1302927947401>, 1 }
    sphere {  m*<7.766720779996052,-0.29442623824278064,-5.449200495305236>, 1 }
    sphere {  m*<-6.056375372319778,5.078760804812143,-3.310602362923289>, 1}
    sphere { m*<-2.2717144446562285,-3.6708888287680965,-1.3476900676460621>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.41136934199607667,-0.2055059622484244,9.1302927947401>, <-1.0220397207861596,-1.0139449825183713,-0.7334384808710246>, 0.5 }
    cylinder { m*<7.766720779996052,-0.29442623824278064,-5.449200495305236>, <-1.0220397207861596,-1.0139449825183713,-0.7334384808710246>, 0.5}
    cylinder { m*<-6.056375372319778,5.078760804812143,-3.310602362923289>, <-1.0220397207861596,-1.0139449825183713,-0.7334384808710246>, 0.5 }
    cylinder {  m*<-2.2717144446562285,-3.6708888287680965,-1.3476900676460621>, <-1.0220397207861596,-1.0139449825183713,-0.7334384808710246>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    