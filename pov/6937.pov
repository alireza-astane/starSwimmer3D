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
    sphere { m*<-0.8619040586749587,-1.2071861975825542,-0.651465674433158>, 1 }        
    sphere {  m*<0.5633028514464773,-0.33159772724770054,9.207717090134386>, 1 }
    sphere {  m*<7.9186542894464536,-0.42051800324205657,-5.371776199910945>, 1 }
    sphere {  m*<-6.728295420863663,5.727021537217684,-3.653562107791122>, 1}
    sphere { m*<-2.0836714694110636,-3.8803018465669155,-1.251538590612986>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.5633028514464773,-0.33159772724770054,9.207717090134386>, <-0.8619040586749587,-1.2071861975825542,-0.651465674433158>, 0.5 }
    cylinder { m*<7.9186542894464536,-0.42051800324205657,-5.371776199910945>, <-0.8619040586749587,-1.2071861975825542,-0.651465674433158>, 0.5}
    cylinder { m*<-6.728295420863663,5.727021537217684,-3.653562107791122>, <-0.8619040586749587,-1.2071861975825542,-0.651465674433158>, 0.5 }
    cylinder {  m*<-2.0836714694110636,-3.8803018465669155,-1.251538590612986>, <-0.8619040586749587,-1.2071861975825542,-0.651465674433158>, 0.5}

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
    sphere { m*<-0.8619040586749587,-1.2071861975825542,-0.651465674433158>, 1 }        
    sphere {  m*<0.5633028514464773,-0.33159772724770054,9.207717090134386>, 1 }
    sphere {  m*<7.9186542894464536,-0.42051800324205657,-5.371776199910945>, 1 }
    sphere {  m*<-6.728295420863663,5.727021537217684,-3.653562107791122>, 1}
    sphere { m*<-2.0836714694110636,-3.8803018465669155,-1.251538590612986>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.5633028514464773,-0.33159772724770054,9.207717090134386>, <-0.8619040586749587,-1.2071861975825542,-0.651465674433158>, 0.5 }
    cylinder { m*<7.9186542894464536,-0.42051800324205657,-5.371776199910945>, <-0.8619040586749587,-1.2071861975825542,-0.651465674433158>, 0.5}
    cylinder { m*<-6.728295420863663,5.727021537217684,-3.653562107791122>, <-0.8619040586749587,-1.2071861975825542,-0.651465674433158>, 0.5 }
    cylinder {  m*<-2.0836714694110636,-3.8803018465669155,-1.251538590612986>, <-0.8619040586749587,-1.2071861975825542,-0.651465674433158>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    