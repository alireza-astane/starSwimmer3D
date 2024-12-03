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
    sphere { m*<1.2562810055084657,0.040616914227555895,0.6086648372278358>, 1 }        
    sphere {  m*<1.5005265213656171,0.043561410055920506,3.598704081032693>, 1 }
    sphere {  m*<3.993773710428154,0.04356141005592051,-0.6185781274579232>, 1 }
    sphere {  m*<-3.6208445371841047,8.024482531891906,-2.2750184563971088>, 1}
    sphere { m*<-3.7011127203587937,-8.134143217207045,-2.3217881297763476>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.5005265213656171,0.043561410055920506,3.598704081032693>, <1.2562810055084657,0.040616914227555895,0.6086648372278358>, 0.5 }
    cylinder { m*<3.993773710428154,0.04356141005592051,-0.6185781274579232>, <1.2562810055084657,0.040616914227555895,0.6086648372278358>, 0.5}
    cylinder { m*<-3.6208445371841047,8.024482531891906,-2.2750184563971088>, <1.2562810055084657,0.040616914227555895,0.6086648372278358>, 0.5 }
    cylinder {  m*<-3.7011127203587937,-8.134143217207045,-2.3217881297763476>, <1.2562810055084657,0.040616914227555895,0.6086648372278358>, 0.5}

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
    sphere { m*<1.2562810055084657,0.040616914227555895,0.6086648372278358>, 1 }        
    sphere {  m*<1.5005265213656171,0.043561410055920506,3.598704081032693>, 1 }
    sphere {  m*<3.993773710428154,0.04356141005592051,-0.6185781274579232>, 1 }
    sphere {  m*<-3.6208445371841047,8.024482531891906,-2.2750184563971088>, 1}
    sphere { m*<-3.7011127203587937,-8.134143217207045,-2.3217881297763476>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.5005265213656171,0.043561410055920506,3.598704081032693>, <1.2562810055084657,0.040616914227555895,0.6086648372278358>, 0.5 }
    cylinder { m*<3.993773710428154,0.04356141005592051,-0.6185781274579232>, <1.2562810055084657,0.040616914227555895,0.6086648372278358>, 0.5}
    cylinder { m*<-3.6208445371841047,8.024482531891906,-2.2750184563971088>, <1.2562810055084657,0.040616914227555895,0.6086648372278358>, 0.5 }
    cylinder {  m*<-3.7011127203587937,-8.134143217207045,-2.3217881297763476>, <1.2562810055084657,0.040616914227555895,0.6086648372278358>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    