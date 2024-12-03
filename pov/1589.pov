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
    sphere { m*<0.7972739004884424,-4.008367584013552e-18,0.8785395597952849>, 1 }        
    sphere {  m*<0.925577227934811,-2.1860128168826937e-19,3.875799396193888>, 1 }
    sphere {  m*<6.24274597749155,4.6991181611551274e-18,-1.3132458005629435>, 1 }
    sphere {  m*<-4.055024808442048,8.164965809277259,-2.250139918678445>, 1}
    sphere { m*<-4.055024808442048,-8.164965809277259,-2.250139918678448>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.925577227934811,-2.1860128168826937e-19,3.875799396193888>, <0.7972739004884424,-4.008367584013552e-18,0.8785395597952849>, 0.5 }
    cylinder { m*<6.24274597749155,4.6991181611551274e-18,-1.3132458005629435>, <0.7972739004884424,-4.008367584013552e-18,0.8785395597952849>, 0.5}
    cylinder { m*<-4.055024808442048,8.164965809277259,-2.250139918678445>, <0.7972739004884424,-4.008367584013552e-18,0.8785395597952849>, 0.5 }
    cylinder {  m*<-4.055024808442048,-8.164965809277259,-2.250139918678448>, <0.7972739004884424,-4.008367584013552e-18,0.8785395597952849>, 0.5}

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
    sphere { m*<0.7972739004884424,-4.008367584013552e-18,0.8785395597952849>, 1 }        
    sphere {  m*<0.925577227934811,-2.1860128168826937e-19,3.875799396193888>, 1 }
    sphere {  m*<6.24274597749155,4.6991181611551274e-18,-1.3132458005629435>, 1 }
    sphere {  m*<-4.055024808442048,8.164965809277259,-2.250139918678445>, 1}
    sphere { m*<-4.055024808442048,-8.164965809277259,-2.250139918678448>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.925577227934811,-2.1860128168826937e-19,3.875799396193888>, <0.7972739004884424,-4.008367584013552e-18,0.8785395597952849>, 0.5 }
    cylinder { m*<6.24274597749155,4.6991181611551274e-18,-1.3132458005629435>, <0.7972739004884424,-4.008367584013552e-18,0.8785395597952849>, 0.5}
    cylinder { m*<-4.055024808442048,8.164965809277259,-2.250139918678445>, <0.7972739004884424,-4.008367584013552e-18,0.8785395597952849>, 0.5 }
    cylinder {  m*<-4.055024808442048,-8.164965809277259,-2.250139918678448>, <0.7972739004884424,-4.008367584013552e-18,0.8785395597952849>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    