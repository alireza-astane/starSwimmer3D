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
    sphere { m*<1.1327127760235947,-2.648056902564161e-19,0.7077622395627893>, 1 }        
    sphere {  m*<1.3377597822538425,1.3449363620138914e-18,3.7007549602499346>, 1 }
    sphere {  m*<4.747002214021187,6.570733983215038e-18,-0.8558880265681201>, 1 }
    sphere {  m*<-3.798151561654138,8.164965809277259,-2.295584942162021>, 1}
    sphere { m*<-3.798151561654138,-8.164965809277259,-2.2955849421620247>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.3377597822538425,1.3449363620138914e-18,3.7007549602499346>, <1.1327127760235947,-2.648056902564161e-19,0.7077622395627893>, 0.5 }
    cylinder { m*<4.747002214021187,6.570733983215038e-18,-0.8558880265681201>, <1.1327127760235947,-2.648056902564161e-19,0.7077622395627893>, 0.5}
    cylinder { m*<-3.798151561654138,8.164965809277259,-2.295584942162021>, <1.1327127760235947,-2.648056902564161e-19,0.7077622395627893>, 0.5 }
    cylinder {  m*<-3.798151561654138,-8.164965809277259,-2.2955849421620247>, <1.1327127760235947,-2.648056902564161e-19,0.7077622395627893>, 0.5}

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
    sphere { m*<1.1327127760235947,-2.648056902564161e-19,0.7077622395627893>, 1 }        
    sphere {  m*<1.3377597822538425,1.3449363620138914e-18,3.7007549602499346>, 1 }
    sphere {  m*<4.747002214021187,6.570733983215038e-18,-0.8558880265681201>, 1 }
    sphere {  m*<-3.798151561654138,8.164965809277259,-2.295584942162021>, 1}
    sphere { m*<-3.798151561654138,-8.164965809277259,-2.2955849421620247>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.3377597822538425,1.3449363620138914e-18,3.7007549602499346>, <1.1327127760235947,-2.648056902564161e-19,0.7077622395627893>, 0.5 }
    cylinder { m*<4.747002214021187,6.570733983215038e-18,-0.8558880265681201>, <1.1327127760235947,-2.648056902564161e-19,0.7077622395627893>, 0.5}
    cylinder { m*<-3.798151561654138,8.164965809277259,-2.295584942162021>, <1.1327127760235947,-2.648056902564161e-19,0.7077622395627893>, 0.5 }
    cylinder {  m*<-3.798151561654138,-8.164965809277259,-2.2955849421620247>, <1.1327127760235947,-2.648056902564161e-19,0.7077622395627893>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    