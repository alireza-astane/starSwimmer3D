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
    sphere { m*<0.16454075102433302,0.518919102095072,-0.0327165843506205>, 1 }        
    sphere {  m*<0.40527585576602465,0.6476291802753975,2.95483818676993>, 1 }
    sphere {  m*<2.8992491450305913,0.6209530774814465,-1.2619261098018058>, 1 }
    sphere {  m*<-1.4570746088685573,2.847393046513673,-1.0066623497665912>, 1}
    sphere { m*<-2.8414089932801767,-5.163397926098273,-1.7743448970065696>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.40527585576602465,0.6476291802753975,2.95483818676993>, <0.16454075102433302,0.518919102095072,-0.0327165843506205>, 0.5 }
    cylinder { m*<2.8992491450305913,0.6209530774814465,-1.2619261098018058>, <0.16454075102433302,0.518919102095072,-0.0327165843506205>, 0.5}
    cylinder { m*<-1.4570746088685573,2.847393046513673,-1.0066623497665912>, <0.16454075102433302,0.518919102095072,-0.0327165843506205>, 0.5 }
    cylinder {  m*<-2.8414089932801767,-5.163397926098273,-1.7743448970065696>, <0.16454075102433302,0.518919102095072,-0.0327165843506205>, 0.5}

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
    sphere { m*<0.16454075102433302,0.518919102095072,-0.0327165843506205>, 1 }        
    sphere {  m*<0.40527585576602465,0.6476291802753975,2.95483818676993>, 1 }
    sphere {  m*<2.8992491450305913,0.6209530774814465,-1.2619261098018058>, 1 }
    sphere {  m*<-1.4570746088685573,2.847393046513673,-1.0066623497665912>, 1}
    sphere { m*<-2.8414089932801767,-5.163397926098273,-1.7743448970065696>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.40527585576602465,0.6476291802753975,2.95483818676993>, <0.16454075102433302,0.518919102095072,-0.0327165843506205>, 0.5 }
    cylinder { m*<2.8992491450305913,0.6209530774814465,-1.2619261098018058>, <0.16454075102433302,0.518919102095072,-0.0327165843506205>, 0.5}
    cylinder { m*<-1.4570746088685573,2.847393046513673,-1.0066623497665912>, <0.16454075102433302,0.518919102095072,-0.0327165843506205>, 0.5 }
    cylinder {  m*<-2.8414089932801767,-5.163397926098273,-1.7743448970065696>, <0.16454075102433302,0.518919102095072,-0.0327165843506205>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    