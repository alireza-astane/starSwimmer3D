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
    sphere { m*<-0.3837678241845314,-0.35631161972229464,-0.4272821071072036>, 1 }        
    sphere {  m*<1.0353996700156296,0.6336272941576222,9.42200798992794>, 1 }
    sphere {  m*<8.403186868338427,0.3485350433653607,-5.148669439145986>, 1 }
    sphere {  m*<-6.492776325350567,6.871616416985995,-3.6578625359643784>, 1}
    sphere { m*<-4.108044164355937,-8.46705829455368,-2.1519481106577283>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.0353996700156296,0.6336272941576222,9.42200798992794>, <-0.3837678241845314,-0.35631161972229464,-0.4272821071072036>, 0.5 }
    cylinder { m*<8.403186868338427,0.3485350433653607,-5.148669439145986>, <-0.3837678241845314,-0.35631161972229464,-0.4272821071072036>, 0.5}
    cylinder { m*<-6.492776325350567,6.871616416985995,-3.6578625359643784>, <-0.3837678241845314,-0.35631161972229464,-0.4272821071072036>, 0.5 }
    cylinder {  m*<-4.108044164355937,-8.46705829455368,-2.1519481106577283>, <-0.3837678241845314,-0.35631161972229464,-0.4272821071072036>, 0.5}

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
    sphere { m*<-0.3837678241845314,-0.35631161972229464,-0.4272821071072036>, 1 }        
    sphere {  m*<1.0353996700156296,0.6336272941576222,9.42200798992794>, 1 }
    sphere {  m*<8.403186868338427,0.3485350433653607,-5.148669439145986>, 1 }
    sphere {  m*<-6.492776325350567,6.871616416985995,-3.6578625359643784>, 1}
    sphere { m*<-4.108044164355937,-8.46705829455368,-2.1519481106577283>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.0353996700156296,0.6336272941576222,9.42200798992794>, <-0.3837678241845314,-0.35631161972229464,-0.4272821071072036>, 0.5 }
    cylinder { m*<8.403186868338427,0.3485350433653607,-5.148669439145986>, <-0.3837678241845314,-0.35631161972229464,-0.4272821071072036>, 0.5}
    cylinder { m*<-6.492776325350567,6.871616416985995,-3.6578625359643784>, <-0.3837678241845314,-0.35631161972229464,-0.4272821071072036>, 0.5 }
    cylinder {  m*<-4.108044164355937,-8.46705829455368,-2.1519481106577283>, <-0.3837678241845314,-0.35631161972229464,-0.4272821071072036>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    