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
    sphere { m*<0.8226966356991017,-2.9590722903061003e-18,0.8666190668109868>, 1 }        
    sphere {  m*<0.9562402609627286,2.519765292999764e-18,3.8636502222277143>, 1 }
    sphere {  m*<6.134541440448403,5.792900076241552e-18,-1.282194354574212>, 1 }
    sphere {  m*<-4.035065288866089,8.164965809277259,-2.2535719880783907>, 1}
    sphere { m*<-4.035065288866089,-8.164965809277259,-2.253571988078394>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.9562402609627286,2.519765292999764e-18,3.8636502222277143>, <0.8226966356991017,-2.9590722903061003e-18,0.8666190668109868>, 0.5 }
    cylinder { m*<6.134541440448403,5.792900076241552e-18,-1.282194354574212>, <0.8226966356991017,-2.9590722903061003e-18,0.8666190668109868>, 0.5}
    cylinder { m*<-4.035065288866089,8.164965809277259,-2.2535719880783907>, <0.8226966356991017,-2.9590722903061003e-18,0.8666190668109868>, 0.5 }
    cylinder {  m*<-4.035065288866089,-8.164965809277259,-2.253571988078394>, <0.8226966356991017,-2.9590722903061003e-18,0.8666190668109868>, 0.5}

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
    sphere { m*<0.8226966356991017,-2.9590722903061003e-18,0.8666190668109868>, 1 }        
    sphere {  m*<0.9562402609627286,2.519765292999764e-18,3.8636502222277143>, 1 }
    sphere {  m*<6.134541440448403,5.792900076241552e-18,-1.282194354574212>, 1 }
    sphere {  m*<-4.035065288866089,8.164965809277259,-2.2535719880783907>, 1}
    sphere { m*<-4.035065288866089,-8.164965809277259,-2.253571988078394>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.9562402609627286,2.519765292999764e-18,3.8636502222277143>, <0.8226966356991017,-2.9590722903061003e-18,0.8666190668109868>, 0.5 }
    cylinder { m*<6.134541440448403,5.792900076241552e-18,-1.282194354574212>, <0.8226966356991017,-2.9590722903061003e-18,0.8666190668109868>, 0.5}
    cylinder { m*<-4.035065288866089,8.164965809277259,-2.2535719880783907>, <0.8226966356991017,-2.9590722903061003e-18,0.8666190668109868>, 0.5 }
    cylinder {  m*<-4.035065288866089,-8.164965809277259,-2.253571988078394>, <0.8226966356991017,-2.9590722903061003e-18,0.8666190668109868>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    