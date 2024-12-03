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
    sphere { m*<-1.4307854068603292,-0.45356510665650607,-0.9430840091753421>, 1 }        
    sphere {  m*<0.02614592048192299,0.11435623013798718,8.933984551166427>, 1 }
    sphere {  m*<7.381497358481895,0.02543595414362998,-5.645508738878929>, 1 }
    sphere {  m*<-4.1703461776141335,3.15466478527975,-2.3473016225468264>, 1}
    sphere { m*<-2.7786557608936215,-3.0498015902221045,-1.6072484034615626>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.02614592048192299,0.11435623013798718,8.933984551166427>, <-1.4307854068603292,-0.45356510665650607,-0.9430840091753421>, 0.5 }
    cylinder { m*<7.381497358481895,0.02543595414362998,-5.645508738878929>, <-1.4307854068603292,-0.45356510665650607,-0.9430840091753421>, 0.5}
    cylinder { m*<-4.1703461776141335,3.15466478527975,-2.3473016225468264>, <-1.4307854068603292,-0.45356510665650607,-0.9430840091753421>, 0.5 }
    cylinder {  m*<-2.7786557608936215,-3.0498015902221045,-1.6072484034615626>, <-1.4307854068603292,-0.45356510665650607,-0.9430840091753421>, 0.5}

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
    sphere { m*<-1.4307854068603292,-0.45356510665650607,-0.9430840091753421>, 1 }        
    sphere {  m*<0.02614592048192299,0.11435623013798718,8.933984551166427>, 1 }
    sphere {  m*<7.381497358481895,0.02543595414362998,-5.645508738878929>, 1 }
    sphere {  m*<-4.1703461776141335,3.15466478527975,-2.3473016225468264>, 1}
    sphere { m*<-2.7786557608936215,-3.0498015902221045,-1.6072484034615626>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.02614592048192299,0.11435623013798718,8.933984551166427>, <-1.4307854068603292,-0.45356510665650607,-0.9430840091753421>, 0.5 }
    cylinder { m*<7.381497358481895,0.02543595414362998,-5.645508738878929>, <-1.4307854068603292,-0.45356510665650607,-0.9430840091753421>, 0.5}
    cylinder { m*<-4.1703461776141335,3.15466478527975,-2.3473016225468264>, <-1.4307854068603292,-0.45356510665650607,-0.9430840091753421>, 0.5 }
    cylinder {  m*<-2.7786557608936215,-3.0498015902221045,-1.6072484034615626>, <-1.4307854068603292,-0.45356510665650607,-0.9430840091753421>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    