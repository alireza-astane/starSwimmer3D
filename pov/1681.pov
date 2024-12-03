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
    sphere { m*<0.9129592939695917,-1.3943283987917392e-18,0.8230552431296401>, 1 }        
    sphere {  m*<1.0658311514962029,1.9869314239008655e-18,3.8191635593665074>, 1 }
    sphere {  m*<5.7448306166821785,5.81495148471577e-18,-1.1680015624386877>, 1 }
    sphere {  m*<-3.9648283423499735,8.164965809277259,-2.2657390480009916>, 1}
    sphere { m*<-3.9648283423499735,-8.164965809277259,-2.265739048000995>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.0658311514962029,1.9869314239008655e-18,3.8191635593665074>, <0.9129592939695917,-1.3943283987917392e-18,0.8230552431296401>, 0.5 }
    cylinder { m*<5.7448306166821785,5.81495148471577e-18,-1.1680015624386877>, <0.9129592939695917,-1.3943283987917392e-18,0.8230552431296401>, 0.5}
    cylinder { m*<-3.9648283423499735,8.164965809277259,-2.2657390480009916>, <0.9129592939695917,-1.3943283987917392e-18,0.8230552431296401>, 0.5 }
    cylinder {  m*<-3.9648283423499735,-8.164965809277259,-2.265739048000995>, <0.9129592939695917,-1.3943283987917392e-18,0.8230552431296401>, 0.5}

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
    sphere { m*<0.9129592939695917,-1.3943283987917392e-18,0.8230552431296401>, 1 }        
    sphere {  m*<1.0658311514962029,1.9869314239008655e-18,3.8191635593665074>, 1 }
    sphere {  m*<5.7448306166821785,5.81495148471577e-18,-1.1680015624386877>, 1 }
    sphere {  m*<-3.9648283423499735,8.164965809277259,-2.2657390480009916>, 1}
    sphere { m*<-3.9648283423499735,-8.164965809277259,-2.265739048000995>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.0658311514962029,1.9869314239008655e-18,3.8191635593665074>, <0.9129592939695917,-1.3943283987917392e-18,0.8230552431296401>, 0.5 }
    cylinder { m*<5.7448306166821785,5.81495148471577e-18,-1.1680015624386877>, <0.9129592939695917,-1.3943283987917392e-18,0.8230552431296401>, 0.5}
    cylinder { m*<-3.9648283423499735,8.164965809277259,-2.2657390480009916>, <0.9129592939695917,-1.3943283987917392e-18,0.8230552431296401>, 0.5 }
    cylinder {  m*<-3.9648283423499735,-8.164965809277259,-2.265739048000995>, <0.9129592939695917,-1.3943283987917392e-18,0.8230552431296401>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    