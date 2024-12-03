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
    sphere { m*<-0.1725213181213246,-0.08745707749513355,-0.4967134614724938>, 1 }        
    sphere {  m*<0.2002168690537276,0.11182902688740323,4.12901717581249>, 1 }
    sphere {  m*<2.5621870758849323,0.014576897891240573,-1.725922986923677>, 1 }
    sphere {  m*<-1.7941366780142147,2.2410168669234656,-1.4706592268884635>, 1}
    sphere { m*<-1.526349456976383,-2.646675075480432,-1.2811129417258909>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.2002168690537276,0.11182902688740323,4.12901717581249>, <-0.1725213181213246,-0.08745707749513355,-0.4967134614724938>, 0.5 }
    cylinder { m*<2.5621870758849323,0.014576897891240573,-1.725922986923677>, <-0.1725213181213246,-0.08745707749513355,-0.4967134614724938>, 0.5}
    cylinder { m*<-1.7941366780142147,2.2410168669234656,-1.4706592268884635>, <-0.1725213181213246,-0.08745707749513355,-0.4967134614724938>, 0.5 }
    cylinder {  m*<-1.526349456976383,-2.646675075480432,-1.2811129417258909>, <-0.1725213181213246,-0.08745707749513355,-0.4967134614724938>, 0.5}

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
    sphere { m*<-0.1725213181213246,-0.08745707749513355,-0.4967134614724938>, 1 }        
    sphere {  m*<0.2002168690537276,0.11182902688740323,4.12901717581249>, 1 }
    sphere {  m*<2.5621870758849323,0.014576897891240573,-1.725922986923677>, 1 }
    sphere {  m*<-1.7941366780142147,2.2410168669234656,-1.4706592268884635>, 1}
    sphere { m*<-1.526349456976383,-2.646675075480432,-1.2811129417258909>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.2002168690537276,0.11182902688740323,4.12901717581249>, <-0.1725213181213246,-0.08745707749513355,-0.4967134614724938>, 0.5 }
    cylinder { m*<2.5621870758849323,0.014576897891240573,-1.725922986923677>, <-0.1725213181213246,-0.08745707749513355,-0.4967134614724938>, 0.5}
    cylinder { m*<-1.7941366780142147,2.2410168669234656,-1.4706592268884635>, <-0.1725213181213246,-0.08745707749513355,-0.4967134614724938>, 0.5 }
    cylinder {  m*<-1.526349456976383,-2.646675075480432,-1.2811129417258909>, <-0.1725213181213246,-0.08745707749513355,-0.4967134614724938>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    