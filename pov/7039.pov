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
    sphere { m*<-0.7880172824890488,-1.2366880070434492,-0.6144849980880137>, 1 }        
    sphere {  m*<0.6311502117111144,-0.246749093163531,9.234805098947144>, 1 }
    sphere {  m*<7.998937410033925,-0.5318413439557927,-5.335872330126799>, 1 }
    sphere {  m*<-6.89702578365508,5.991240029664864,-3.845065426945192>, 1}
    sphere { m*<-2.13187057135178,-4.16333807232914,-1.2368067243416698>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.6311502117111144,-0.246749093163531,9.234805098947144>, <-0.7880172824890488,-1.2366880070434492,-0.6144849980880137>, 0.5 }
    cylinder { m*<7.998937410033925,-0.5318413439557927,-5.335872330126799>, <-0.7880172824890488,-1.2366880070434492,-0.6144849980880137>, 0.5}
    cylinder { m*<-6.89702578365508,5.991240029664864,-3.845065426945192>, <-0.7880172824890488,-1.2366880070434492,-0.6144849980880137>, 0.5 }
    cylinder {  m*<-2.13187057135178,-4.16333807232914,-1.2368067243416698>, <-0.7880172824890488,-1.2366880070434492,-0.6144849980880137>, 0.5}

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
    sphere { m*<-0.7880172824890488,-1.2366880070434492,-0.6144849980880137>, 1 }        
    sphere {  m*<0.6311502117111144,-0.246749093163531,9.234805098947144>, 1 }
    sphere {  m*<7.998937410033925,-0.5318413439557927,-5.335872330126799>, 1 }
    sphere {  m*<-6.89702578365508,5.991240029664864,-3.845065426945192>, 1}
    sphere { m*<-2.13187057135178,-4.16333807232914,-1.2368067243416698>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.6311502117111144,-0.246749093163531,9.234805098947144>, <-0.7880172824890488,-1.2366880070434492,-0.6144849980880137>, 0.5 }
    cylinder { m*<7.998937410033925,-0.5318413439557927,-5.335872330126799>, <-0.7880172824890488,-1.2366880070434492,-0.6144849980880137>, 0.5}
    cylinder { m*<-6.89702578365508,5.991240029664864,-3.845065426945192>, <-0.7880172824890488,-1.2366880070434492,-0.6144849980880137>, 0.5 }
    cylinder {  m*<-2.13187057135178,-4.16333807232914,-1.2368067243416698>, <-0.7880172824890488,-1.2366880070434492,-0.6144849980880137>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    