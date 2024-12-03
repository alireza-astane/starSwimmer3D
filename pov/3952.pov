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
    sphere { m*<-0.12667824082575452,-0.03158864989195154,-0.20144703047675294>, 1 }        
    sphere {  m*<0.11405686391593689,0.09712142828837389,2.786107740643797>, 1 }
    sphere {  m*<2.6080301531805072,0.07044532549442284,-1.4306565559279383>, 1 }
    sphere {  m*<-1.7482936007186467,2.296885294526651,-1.1753927958927242>, 1}
    sphere { m*<-1.6289761989085747,-2.8714675549896214,-1.0718689870645957>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.11405686391593689,0.09712142828837389,2.786107740643797>, <-0.12667824082575452,-0.03158864989195154,-0.20144703047675294>, 0.5 }
    cylinder { m*<2.6080301531805072,0.07044532549442284,-1.4306565559279383>, <-0.12667824082575452,-0.03158864989195154,-0.20144703047675294>, 0.5}
    cylinder { m*<-1.7482936007186467,2.296885294526651,-1.1753927958927242>, <-0.12667824082575452,-0.03158864989195154,-0.20144703047675294>, 0.5 }
    cylinder {  m*<-1.6289761989085747,-2.8714675549896214,-1.0718689870645957>, <-0.12667824082575452,-0.03158864989195154,-0.20144703047675294>, 0.5}

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
    sphere { m*<-0.12667824082575452,-0.03158864989195154,-0.20144703047675294>, 1 }        
    sphere {  m*<0.11405686391593689,0.09712142828837389,2.786107740643797>, 1 }
    sphere {  m*<2.6080301531805072,0.07044532549442284,-1.4306565559279383>, 1 }
    sphere {  m*<-1.7482936007186467,2.296885294526651,-1.1753927958927242>, 1}
    sphere { m*<-1.6289761989085747,-2.8714675549896214,-1.0718689870645957>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.11405686391593689,0.09712142828837389,2.786107740643797>, <-0.12667824082575452,-0.03158864989195154,-0.20144703047675294>, 0.5 }
    cylinder { m*<2.6080301531805072,0.07044532549442284,-1.4306565559279383>, <-0.12667824082575452,-0.03158864989195154,-0.20144703047675294>, 0.5}
    cylinder { m*<-1.7482936007186467,2.296885294526651,-1.1753927958927242>, <-0.12667824082575452,-0.03158864989195154,-0.20144703047675294>, 0.5 }
    cylinder {  m*<-1.6289761989085747,-2.8714675549896214,-1.0718689870645957>, <-0.12667824082575452,-0.03158864989195154,-0.20144703047675294>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    