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
    sphere { m*<-0.020755105709433064,0.16864385083149291,-0.14007583442352362>, 1 }        
    sphere {  m*<0.21997999903225862,0.29735392901181845,2.8474789366970277>, 1 }
    sphere {  m*<2.7139532882968296,0.2706778262178676,-1.3692853598747106>, 1 }
    sphere {  m*<-1.6423704656023248,2.4971177952500954,-1.114021599839496>, 1}
    sphere { m*<-2.1232502053513285,-3.8058217000396635,-1.3582482272570413>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.21997999903225862,0.29735392901181845,2.8474789366970277>, <-0.020755105709433064,0.16864385083149291,-0.14007583442352362>, 0.5 }
    cylinder { m*<2.7139532882968296,0.2706778262178676,-1.3692853598747106>, <-0.020755105709433064,0.16864385083149291,-0.14007583442352362>, 0.5}
    cylinder { m*<-1.6423704656023248,2.4971177952500954,-1.114021599839496>, <-0.020755105709433064,0.16864385083149291,-0.14007583442352362>, 0.5 }
    cylinder {  m*<-2.1232502053513285,-3.8058217000396635,-1.3582482272570413>, <-0.020755105709433064,0.16864385083149291,-0.14007583442352362>, 0.5}

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
    sphere { m*<-0.020755105709433064,0.16864385083149291,-0.14007583442352362>, 1 }        
    sphere {  m*<0.21997999903225862,0.29735392901181845,2.8474789366970277>, 1 }
    sphere {  m*<2.7139532882968296,0.2706778262178676,-1.3692853598747106>, 1 }
    sphere {  m*<-1.6423704656023248,2.4971177952500954,-1.114021599839496>, 1}
    sphere { m*<-2.1232502053513285,-3.8058217000396635,-1.3582482272570413>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.21997999903225862,0.29735392901181845,2.8474789366970277>, <-0.020755105709433064,0.16864385083149291,-0.14007583442352362>, 0.5 }
    cylinder { m*<2.7139532882968296,0.2706778262178676,-1.3692853598747106>, <-0.020755105709433064,0.16864385083149291,-0.14007583442352362>, 0.5}
    cylinder { m*<-1.6423704656023248,2.4971177952500954,-1.114021599839496>, <-0.020755105709433064,0.16864385083149291,-0.14007583442352362>, 0.5 }
    cylinder {  m*<-2.1232502053513285,-3.8058217000396635,-1.3582482272570413>, <-0.020755105709433064,0.16864385083149291,-0.14007583442352362>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    