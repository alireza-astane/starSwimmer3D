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
    sphere { m*<0.6301161871705016,-6.1249577595525836e-18,0.9534837608881278>, 1 }        
    sphere {  m*<0.7260343563347257,-2.6872197774415953e-18,3.9519533421723665>, 1 }
    sphere {  m*<6.9402567678932305,1.6218734948217657e-18,-1.5073713646827962>, 1 }
    sphere {  m*<-4.188099943251245,8.164965809277259,-2.2274282453054273>, 1}
    sphere { m*<-4.188099943251245,-8.164965809277259,-2.22742824530543>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.7260343563347257,-2.6872197774415953e-18,3.9519533421723665>, <0.6301161871705016,-6.1249577595525836e-18,0.9534837608881278>, 0.5 }
    cylinder { m*<6.9402567678932305,1.6218734948217657e-18,-1.5073713646827962>, <0.6301161871705016,-6.1249577595525836e-18,0.9534837608881278>, 0.5}
    cylinder { m*<-4.188099943251245,8.164965809277259,-2.2274282453054273>, <0.6301161871705016,-6.1249577595525836e-18,0.9534837608881278>, 0.5 }
    cylinder {  m*<-4.188099943251245,-8.164965809277259,-2.22742824530543>, <0.6301161871705016,-6.1249577595525836e-18,0.9534837608881278>, 0.5}

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
    sphere { m*<0.6301161871705016,-6.1249577595525836e-18,0.9534837608881278>, 1 }        
    sphere {  m*<0.7260343563347257,-2.6872197774415953e-18,3.9519533421723665>, 1 }
    sphere {  m*<6.9402567678932305,1.6218734948217657e-18,-1.5073713646827962>, 1 }
    sphere {  m*<-4.188099943251245,8.164965809277259,-2.2274282453054273>, 1}
    sphere { m*<-4.188099943251245,-8.164965809277259,-2.22742824530543>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.7260343563347257,-2.6872197774415953e-18,3.9519533421723665>, <0.6301161871705016,-6.1249577595525836e-18,0.9534837608881278>, 0.5 }
    cylinder { m*<6.9402567678932305,1.6218734948217657e-18,-1.5073713646827962>, <0.6301161871705016,-6.1249577595525836e-18,0.9534837608881278>, 0.5}
    cylinder { m*<-4.188099943251245,8.164965809277259,-2.2274282453054273>, <0.6301161871705016,-6.1249577595525836e-18,0.9534837608881278>, 0.5 }
    cylinder {  m*<-4.188099943251245,-8.164965809277259,-2.22742824530543>, <0.6301161871705016,-6.1249577595525836e-18,0.9534837608881278>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    