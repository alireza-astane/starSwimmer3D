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
    sphere { m*<1.050038951424416,-6.2324506067647015e-19,0.7528094880639251>, 1 }        
    sphere {  m*<1.234561236624529,1.4005976175974732e-18,3.7471366693409065>, 1 }
    sphere {  m*<5.132463591910908,5.5723320136106684e-18,-0.9801846754338766>, 1 }
    sphere {  m*<-3.8601170431804968,8.164965809277259,-2.2842719152548607>, 1}
    sphere { m*<-3.8601170431804968,-8.164965809277259,-2.2842719152548634>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.234561236624529,1.4005976175974732e-18,3.7471366693409065>, <1.050038951424416,-6.2324506067647015e-19,0.7528094880639251>, 0.5 }
    cylinder { m*<5.132463591910908,5.5723320136106684e-18,-0.9801846754338766>, <1.050038951424416,-6.2324506067647015e-19,0.7528094880639251>, 0.5}
    cylinder { m*<-3.8601170431804968,8.164965809277259,-2.2842719152548607>, <1.050038951424416,-6.2324506067647015e-19,0.7528094880639251>, 0.5 }
    cylinder {  m*<-3.8601170431804968,-8.164965809277259,-2.2842719152548634>, <1.050038951424416,-6.2324506067647015e-19,0.7528094880639251>, 0.5}

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
    sphere { m*<1.050038951424416,-6.2324506067647015e-19,0.7528094880639251>, 1 }        
    sphere {  m*<1.234561236624529,1.4005976175974732e-18,3.7471366693409065>, 1 }
    sphere {  m*<5.132463591910908,5.5723320136106684e-18,-0.9801846754338766>, 1 }
    sphere {  m*<-3.8601170431804968,8.164965809277259,-2.2842719152548607>, 1}
    sphere { m*<-3.8601170431804968,-8.164965809277259,-2.2842719152548634>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.234561236624529,1.4005976175974732e-18,3.7471366693409065>, <1.050038951424416,-6.2324506067647015e-19,0.7528094880639251>, 0.5 }
    cylinder { m*<5.132463591910908,5.5723320136106684e-18,-0.9801846754338766>, <1.050038951424416,-6.2324506067647015e-19,0.7528094880639251>, 0.5}
    cylinder { m*<-3.8601170431804968,8.164965809277259,-2.2842719152548607>, <1.050038951424416,-6.2324506067647015e-19,0.7528094880639251>, 0.5 }
    cylinder {  m*<-3.8601170431804968,-8.164965809277259,-2.2842719152548634>, <1.050038951424416,-6.2324506067647015e-19,0.7528094880639251>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    