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
    sphere { m*<-0.6498938410061346,-0.9358821211750947,-0.5505217520609966>, 1 }        
    sphere {  m*<0.7692736531940277,0.054056792704823,9.298768344974155>, 1 }
    sphere {  m*<8.137060851516823,-0.23103545808743986,-5.271909084099777>, 1 }
    sphere {  m*<-6.758902342172162,6.292045915533206,-3.7811021809181726>, 1}
    sphere { m*<-2.8598769629956386,-5.748793844391869,-1.5739374201744767>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.7692736531940277,0.054056792704823,9.298768344974155>, <-0.6498938410061346,-0.9358821211750947,-0.5505217520609966>, 0.5 }
    cylinder { m*<8.137060851516823,-0.23103545808743986,-5.271909084099777>, <-0.6498938410061346,-0.9358821211750947,-0.5505217520609966>, 0.5}
    cylinder { m*<-6.758902342172162,6.292045915533206,-3.7811021809181726>, <-0.6498938410061346,-0.9358821211750947,-0.5505217520609966>, 0.5 }
    cylinder {  m*<-2.8598769629956386,-5.748793844391869,-1.5739374201744767>, <-0.6498938410061346,-0.9358821211750947,-0.5505217520609966>, 0.5}

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
    sphere { m*<-0.6498938410061346,-0.9358821211750947,-0.5505217520609966>, 1 }        
    sphere {  m*<0.7692736531940277,0.054056792704823,9.298768344974155>, 1 }
    sphere {  m*<8.137060851516823,-0.23103545808743986,-5.271909084099777>, 1 }
    sphere {  m*<-6.758902342172162,6.292045915533206,-3.7811021809181726>, 1}
    sphere { m*<-2.8598769629956386,-5.748793844391869,-1.5739374201744767>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.7692736531940277,0.054056792704823,9.298768344974155>, <-0.6498938410061346,-0.9358821211750947,-0.5505217520609966>, 0.5 }
    cylinder { m*<8.137060851516823,-0.23103545808743986,-5.271909084099777>, <-0.6498938410061346,-0.9358821211750947,-0.5505217520609966>, 0.5}
    cylinder { m*<-6.758902342172162,6.292045915533206,-3.7811021809181726>, <-0.6498938410061346,-0.9358821211750947,-0.5505217520609966>, 0.5 }
    cylinder {  m*<-2.8598769629956386,-5.748793844391869,-1.5739374201744767>, <-0.6498938410061346,-0.9358821211750947,-0.5505217520609966>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    