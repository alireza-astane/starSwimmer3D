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
    sphere { m*<0.6550559448144645,-6.142558633477919e-18,0.942657232793972>, 1 }        
    sphere {  m*<0.7555877614427668,-2.1864804938252076e-18,3.9409758616963337>, 1 }
    sphere {  m*<6.8375630248971095,2.2848702934067905e-18,-1.4794007577108934>, 1 }
    sphere {  m*<-4.168049261783232,8.164965809277259,-2.230837400957822>, 1}
    sphere { m*<-4.168049261783232,-8.164965809277259,-2.2308374009578245>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.7555877614427668,-2.1864804938252076e-18,3.9409758616963337>, <0.6550559448144645,-6.142558633477919e-18,0.942657232793972>, 0.5 }
    cylinder { m*<6.8375630248971095,2.2848702934067905e-18,-1.4794007577108934>, <0.6550559448144645,-6.142558633477919e-18,0.942657232793972>, 0.5}
    cylinder { m*<-4.168049261783232,8.164965809277259,-2.230837400957822>, <0.6550559448144645,-6.142558633477919e-18,0.942657232793972>, 0.5 }
    cylinder {  m*<-4.168049261783232,-8.164965809277259,-2.2308374009578245>, <0.6550559448144645,-6.142558633477919e-18,0.942657232793972>, 0.5}

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
    sphere { m*<0.6550559448144645,-6.142558633477919e-18,0.942657232793972>, 1 }        
    sphere {  m*<0.7555877614427668,-2.1864804938252076e-18,3.9409758616963337>, 1 }
    sphere {  m*<6.8375630248971095,2.2848702934067905e-18,-1.4794007577108934>, 1 }
    sphere {  m*<-4.168049261783232,8.164965809277259,-2.230837400957822>, 1}
    sphere { m*<-4.168049261783232,-8.164965809277259,-2.2308374009578245>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.7555877614427668,-2.1864804938252076e-18,3.9409758616963337>, <0.6550559448144645,-6.142558633477919e-18,0.942657232793972>, 0.5 }
    cylinder { m*<6.8375630248971095,2.2848702934067905e-18,-1.4794007577108934>, <0.6550559448144645,-6.142558633477919e-18,0.942657232793972>, 0.5}
    cylinder { m*<-4.168049261783232,8.164965809277259,-2.230837400957822>, <0.6550559448144645,-6.142558633477919e-18,0.942657232793972>, 0.5 }
    cylinder {  m*<-4.168049261783232,-8.164965809277259,-2.2308374009578245>, <0.6550559448144645,-6.142558633477919e-18,0.942657232793972>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    