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
    sphere { m*<0.8542678259358842,-2.411777968600221e-18,0.8516067155471152>, 1 }        
    sphere {  m*<0.994441773249258,2.00075938732511e-18,3.848335377440863>, 1 }
    sphere {  m*<5.999257807524128,5.3277448911352416e-18,-1.2429824261311502>, 1 }
    sphere {  m*<-4.010385800414883,8.164965809277259,-2.2578296894660026>, 1}
    sphere { m*<-4.010385800414883,-8.164965809277259,-2.2578296894660053>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.994441773249258,2.00075938732511e-18,3.848335377440863>, <0.8542678259358842,-2.411777968600221e-18,0.8516067155471152>, 0.5 }
    cylinder { m*<5.999257807524128,5.3277448911352416e-18,-1.2429824261311502>, <0.8542678259358842,-2.411777968600221e-18,0.8516067155471152>, 0.5}
    cylinder { m*<-4.010385800414883,8.164965809277259,-2.2578296894660026>, <0.8542678259358842,-2.411777968600221e-18,0.8516067155471152>, 0.5 }
    cylinder {  m*<-4.010385800414883,-8.164965809277259,-2.2578296894660053>, <0.8542678259358842,-2.411777968600221e-18,0.8516067155471152>, 0.5}

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
    sphere { m*<0.8542678259358842,-2.411777968600221e-18,0.8516067155471152>, 1 }        
    sphere {  m*<0.994441773249258,2.00075938732511e-18,3.848335377440863>, 1 }
    sphere {  m*<5.999257807524128,5.3277448911352416e-18,-1.2429824261311502>, 1 }
    sphere {  m*<-4.010385800414883,8.164965809277259,-2.2578296894660026>, 1}
    sphere { m*<-4.010385800414883,-8.164965809277259,-2.2578296894660053>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.994441773249258,2.00075938732511e-18,3.848335377440863>, <0.8542678259358842,-2.411777968600221e-18,0.8516067155471152>, 0.5 }
    cylinder { m*<5.999257807524128,5.3277448911352416e-18,-1.2429824261311502>, <0.8542678259358842,-2.411777968600221e-18,0.8516067155471152>, 0.5}
    cylinder { m*<-4.010385800414883,8.164965809277259,-2.2578296894660026>, <0.8542678259358842,-2.411777968600221e-18,0.8516067155471152>, 0.5 }
    cylinder {  m*<-4.010385800414883,-8.164965809277259,-2.2578296894660053>, <0.8542678259358842,-2.411777968600221e-18,0.8516067155471152>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    