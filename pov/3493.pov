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
    sphere { m*<0.15290013732616076,0.4969141908150103,-0.0394610824686748>, 1 }        
    sphere {  m*<0.3936352420678524,0.6256242689953359,2.9480936886518756>, 1 }
    sphere {  m*<2.8876085313324187,0.5989481662013847,-1.26867060791986>, 1 }
    sphere {  m*<-1.46871522256673,2.825388135233612,-1.0134068478846454>, 1}
    sphere { m*<-2.7993477574704317,-5.0838871901249165,-1.7499748824079062>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.3936352420678524,0.6256242689953359,2.9480936886518756>, <0.15290013732616076,0.4969141908150103,-0.0394610824686748>, 0.5 }
    cylinder { m*<2.8876085313324187,0.5989481662013847,-1.26867060791986>, <0.15290013732616076,0.4969141908150103,-0.0394610824686748>, 0.5}
    cylinder { m*<-1.46871522256673,2.825388135233612,-1.0134068478846454>, <0.15290013732616076,0.4969141908150103,-0.0394610824686748>, 0.5 }
    cylinder {  m*<-2.7993477574704317,-5.0838871901249165,-1.7499748824079062>, <0.15290013732616076,0.4969141908150103,-0.0394610824686748>, 0.5}

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
    sphere { m*<0.15290013732616076,0.4969141908150103,-0.0394610824686748>, 1 }        
    sphere {  m*<0.3936352420678524,0.6256242689953359,2.9480936886518756>, 1 }
    sphere {  m*<2.8876085313324187,0.5989481662013847,-1.26867060791986>, 1 }
    sphere {  m*<-1.46871522256673,2.825388135233612,-1.0134068478846454>, 1}
    sphere { m*<-2.7993477574704317,-5.0838871901249165,-1.7499748824079062>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.3936352420678524,0.6256242689953359,2.9480936886518756>, <0.15290013732616076,0.4969141908150103,-0.0394610824686748>, 0.5 }
    cylinder { m*<2.8876085313324187,0.5989481662013847,-1.26867060791986>, <0.15290013732616076,0.4969141908150103,-0.0394610824686748>, 0.5}
    cylinder { m*<-1.46871522256673,2.825388135233612,-1.0134068478846454>, <0.15290013732616076,0.4969141908150103,-0.0394610824686748>, 0.5 }
    cylinder {  m*<-2.7993477574704317,-5.0838871901249165,-1.7499748824079062>, <0.15290013732616076,0.4969141908150103,-0.0394610824686748>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    