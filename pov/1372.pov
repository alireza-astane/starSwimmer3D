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
    sphere { m*<0.5133349576850564,-4.048332784969324e-18,1.0026890482867759>, 1 }        
    sphere {  m*<0.5885930293874351,1.308871338920549e-19,4.001747482530008>, 1 }
    sphere {  m*<7.415725717782578,4.034903997695779e-18,-1.6344398550923336>, 1 }
    sphere {  m*<-4.282850802046699,8.164965809277259,-2.2113414501173185>, 1}
    sphere { m*<-4.282850802046699,-8.164965809277259,-2.211341450117321>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.5885930293874351,1.308871338920549e-19,4.001747482530008>, <0.5133349576850564,-4.048332784969324e-18,1.0026890482867759>, 0.5 }
    cylinder { m*<7.415725717782578,4.034903997695779e-18,-1.6344398550923336>, <0.5133349576850564,-4.048332784969324e-18,1.0026890482867759>, 0.5}
    cylinder { m*<-4.282850802046699,8.164965809277259,-2.2113414501173185>, <0.5133349576850564,-4.048332784969324e-18,1.0026890482867759>, 0.5 }
    cylinder {  m*<-4.282850802046699,-8.164965809277259,-2.211341450117321>, <0.5133349576850564,-4.048332784969324e-18,1.0026890482867759>, 0.5}

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
    sphere { m*<0.5133349576850564,-4.048332784969324e-18,1.0026890482867759>, 1 }        
    sphere {  m*<0.5885930293874351,1.308871338920549e-19,4.001747482530008>, 1 }
    sphere {  m*<7.415725717782578,4.034903997695779e-18,-1.6344398550923336>, 1 }
    sphere {  m*<-4.282850802046699,8.164965809277259,-2.2113414501173185>, 1}
    sphere { m*<-4.282850802046699,-8.164965809277259,-2.211341450117321>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.5885930293874351,1.308871338920549e-19,4.001747482530008>, <0.5133349576850564,-4.048332784969324e-18,1.0026890482867759>, 0.5 }
    cylinder { m*<7.415725717782578,4.034903997695779e-18,-1.6344398550923336>, <0.5133349576850564,-4.048332784969324e-18,1.0026890482867759>, 0.5}
    cylinder { m*<-4.282850802046699,8.164965809277259,-2.2113414501173185>, <0.5133349576850564,-4.048332784969324e-18,1.0026890482867759>, 0.5 }
    cylinder {  m*<-4.282850802046699,-8.164965809277259,-2.211341450117321>, <0.5133349576850564,-4.048332784969324e-18,1.0026890482867759>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    