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
    sphere { m*<0.17899947319129833,-4.110016288332531e-18,1.132060560437366>, 1 }        
    sphere {  m*<0.20269501256776606,-3.2257934872472764e-18,4.131967710817727>, 1 }
    sphere {  m*<8.738080197718983,3.272266649089129e-18,-1.970357462966852>, 1 }
    sphere {  m*<-4.561228212702484,8.164965809277259,-2.163935658703097>, 1}
    sphere { m*<-4.561228212702484,-8.164965809277259,-2.1639356587031005>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.20269501256776606,-3.2257934872472764e-18,4.131967710817727>, <0.17899947319129833,-4.110016288332531e-18,1.132060560437366>, 0.5 }
    cylinder { m*<8.738080197718983,3.272266649089129e-18,-1.970357462966852>, <0.17899947319129833,-4.110016288332531e-18,1.132060560437366>, 0.5}
    cylinder { m*<-4.561228212702484,8.164965809277259,-2.163935658703097>, <0.17899947319129833,-4.110016288332531e-18,1.132060560437366>, 0.5 }
    cylinder {  m*<-4.561228212702484,-8.164965809277259,-2.1639356587031005>, <0.17899947319129833,-4.110016288332531e-18,1.132060560437366>, 0.5}

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
    sphere { m*<0.17899947319129833,-4.110016288332531e-18,1.132060560437366>, 1 }        
    sphere {  m*<0.20269501256776606,-3.2257934872472764e-18,4.131967710817727>, 1 }
    sphere {  m*<8.738080197718983,3.272266649089129e-18,-1.970357462966852>, 1 }
    sphere {  m*<-4.561228212702484,8.164965809277259,-2.163935658703097>, 1}
    sphere { m*<-4.561228212702484,-8.164965809277259,-2.1639356587031005>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.20269501256776606,-3.2257934872472764e-18,4.131967710817727>, <0.17899947319129833,-4.110016288332531e-18,1.132060560437366>, 0.5 }
    cylinder { m*<8.738080197718983,3.272266649089129e-18,-1.970357462966852>, <0.17899947319129833,-4.110016288332531e-18,1.132060560437366>, 0.5}
    cylinder { m*<-4.561228212702484,8.164965809277259,-2.163935658703097>, <0.17899947319129833,-4.110016288332531e-18,1.132060560437366>, 0.5 }
    cylinder {  m*<-4.561228212702484,-8.164965809277259,-2.1639356587031005>, <0.17899947319129833,-4.110016288332531e-18,1.132060560437366>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    