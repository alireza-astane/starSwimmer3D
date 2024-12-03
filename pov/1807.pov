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
    sphere { m*<1.0653955502093193,-6.786099257031919e-19,0.7446033263182632>, 1 }        
    sphere {  m*<1.2536460370946318,1.246737367049982e-18,3.7386986103375888>, 1 }
    sphere {  m*<5.061935781914415,5.8164817611648715e-18,-0.9578182390089911>, 1 }
    sphere {  m*<-3.8485384993270877,8.164965809277259,-2.2863621096181594>, 1}
    sphere { m*<-3.8485384993270877,-8.164965809277259,-2.286362109618163>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.2536460370946318,1.246737367049982e-18,3.7386986103375888>, <1.0653955502093193,-6.786099257031919e-19,0.7446033263182632>, 0.5 }
    cylinder { m*<5.061935781914415,5.8164817611648715e-18,-0.9578182390089911>, <1.0653955502093193,-6.786099257031919e-19,0.7446033263182632>, 0.5}
    cylinder { m*<-3.8485384993270877,8.164965809277259,-2.2863621096181594>, <1.0653955502093193,-6.786099257031919e-19,0.7446033263182632>, 0.5 }
    cylinder {  m*<-3.8485384993270877,-8.164965809277259,-2.286362109618163>, <1.0653955502093193,-6.786099257031919e-19,0.7446033263182632>, 0.5}

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
    sphere { m*<1.0653955502093193,-6.786099257031919e-19,0.7446033263182632>, 1 }        
    sphere {  m*<1.2536460370946318,1.246737367049982e-18,3.7386986103375888>, 1 }
    sphere {  m*<5.061935781914415,5.8164817611648715e-18,-0.9578182390089911>, 1 }
    sphere {  m*<-3.8485384993270877,8.164965809277259,-2.2863621096181594>, 1}
    sphere { m*<-3.8485384993270877,-8.164965809277259,-2.286362109618163>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.2536460370946318,1.246737367049982e-18,3.7386986103375888>, <1.0653955502093193,-6.786099257031919e-19,0.7446033263182632>, 0.5 }
    cylinder { m*<5.061935781914415,5.8164817611648715e-18,-0.9578182390089911>, <1.0653955502093193,-6.786099257031919e-19,0.7446033263182632>, 0.5}
    cylinder { m*<-3.8485384993270877,8.164965809277259,-2.2863621096181594>, <1.0653955502093193,-6.786099257031919e-19,0.7446033263182632>, 0.5 }
    cylinder {  m*<-3.8485384993270877,-8.164965809277259,-2.286362109618163>, <1.0653955502093193,-6.786099257031919e-19,0.7446033263182632>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    