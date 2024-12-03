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
    sphere { m*<0.5253694028322254,-4.780589176697601e-18,0.9977266093730526>, 1 }        
    sphere {  m*<0.6026871316243321,-5.232847157470234e-19,3.996732729712898>, 1 }
    sphere {  m*<7.3671104082491174,3.315419494870264e-18,-1.6216215750932543>, 1 }
    sphere {  m*<-4.273022661408483,8.164965809277259,-2.2130093235937993>, 1}
    sphere { m*<-4.273022661408483,-8.164965809277259,-2.213009323593803>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.6026871316243321,-5.232847157470234e-19,3.996732729712898>, <0.5253694028322254,-4.780589176697601e-18,0.9977266093730526>, 0.5 }
    cylinder { m*<7.3671104082491174,3.315419494870264e-18,-1.6216215750932543>, <0.5253694028322254,-4.780589176697601e-18,0.9977266093730526>, 0.5}
    cylinder { m*<-4.273022661408483,8.164965809277259,-2.2130093235937993>, <0.5253694028322254,-4.780589176697601e-18,0.9977266093730526>, 0.5 }
    cylinder {  m*<-4.273022661408483,-8.164965809277259,-2.213009323593803>, <0.5253694028322254,-4.780589176697601e-18,0.9977266093730526>, 0.5}

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
    sphere { m*<0.5253694028322254,-4.780589176697601e-18,0.9977266093730526>, 1 }        
    sphere {  m*<0.6026871316243321,-5.232847157470234e-19,3.996732729712898>, 1 }
    sphere {  m*<7.3671104082491174,3.315419494870264e-18,-1.6216215750932543>, 1 }
    sphere {  m*<-4.273022661408483,8.164965809277259,-2.2130093235937993>, 1}
    sphere { m*<-4.273022661408483,-8.164965809277259,-2.213009323593803>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.6026871316243321,-5.232847157470234e-19,3.996732729712898>, <0.5253694028322254,-4.780589176697601e-18,0.9977266093730526>, 0.5 }
    cylinder { m*<7.3671104082491174,3.315419494870264e-18,-1.6216215750932543>, <0.5253694028322254,-4.780589176697601e-18,0.9977266093730526>, 0.5}
    cylinder { m*<-4.273022661408483,8.164965809277259,-2.2130093235937993>, <0.5253694028322254,-4.780589176697601e-18,0.9977266093730526>, 0.5 }
    cylinder {  m*<-4.273022661408483,-8.164965809277259,-2.213009323593803>, <0.5253694028322254,-4.780589176697601e-18,0.9977266093730526>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    