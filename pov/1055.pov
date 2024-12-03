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
    sphere { m*<0.07862834767450715,-5.339076241437961e-18,1.1681529125599235>, 1 }        
    sphere {  m*<0.08873834741393191,-5.4209288916574125e-18,4.168136181405744>, 1 }
    sphere {  m*<9.126286366758428,5.0301310456001025e-19,-2.0650448653980362>, 1 }
    sphere {  m*<-4.646623902807224,8.164965809277259,-2.149275498660896>, 1}
    sphere { m*<-4.646623902807224,-8.164965809277259,-2.1492754986608995>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.08873834741393191,-5.4209288916574125e-18,4.168136181405744>, <0.07862834767450715,-5.339076241437961e-18,1.1681529125599235>, 0.5 }
    cylinder { m*<9.126286366758428,5.0301310456001025e-19,-2.0650448653980362>, <0.07862834767450715,-5.339076241437961e-18,1.1681529125599235>, 0.5}
    cylinder { m*<-4.646623902807224,8.164965809277259,-2.149275498660896>, <0.07862834767450715,-5.339076241437961e-18,1.1681529125599235>, 0.5 }
    cylinder {  m*<-4.646623902807224,-8.164965809277259,-2.1492754986608995>, <0.07862834767450715,-5.339076241437961e-18,1.1681529125599235>, 0.5}

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
    sphere { m*<0.07862834767450715,-5.339076241437961e-18,1.1681529125599235>, 1 }        
    sphere {  m*<0.08873834741393191,-5.4209288916574125e-18,4.168136181405744>, 1 }
    sphere {  m*<9.126286366758428,5.0301310456001025e-19,-2.0650448653980362>, 1 }
    sphere {  m*<-4.646623902807224,8.164965809277259,-2.149275498660896>, 1}
    sphere { m*<-4.646623902807224,-8.164965809277259,-2.1492754986608995>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.08873834741393191,-5.4209288916574125e-18,4.168136181405744>, <0.07862834767450715,-5.339076241437961e-18,1.1681529125599235>, 0.5 }
    cylinder { m*<9.126286366758428,5.0301310456001025e-19,-2.0650448653980362>, <0.07862834767450715,-5.339076241437961e-18,1.1681529125599235>, 0.5}
    cylinder { m*<-4.646623902807224,8.164965809277259,-2.149275498660896>, <0.07862834767450715,-5.339076241437961e-18,1.1681529125599235>, 0.5 }
    cylinder {  m*<-4.646623902807224,-8.164965809277259,-2.1492754986608995>, <0.07862834767450715,-5.339076241437961e-18,1.1681529125599235>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    